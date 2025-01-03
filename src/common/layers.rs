use candle_core::{
    DType, Device, Tensor, D
};
use std::fs::File;
use candle_nn::{
    embedding, layer_norm, linear, loss::cross_entropy, ops::softmax, Activation, AdamW, Embedding, LayerNorm, LayerNormConfig, Linear, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap
};
use rand::{thread_rng, Rng};

// トークンの種類
const VOCABS: [char; 7] = ['_', ' ', '0', '1', '+', '=', '.'];

// ハイパーパラメータ
#[derive(Debug, Clone, Copy)]
pub struct HyperParams {
    // 各ベクトルの次元数
    d_model: usize,
    // Q, K, V行列をかけたあとの行列の次元数
    d_head: usize,
    // Multi-Head Attentionのheadの数
    n_head: usize,
    // トークン数
    n_ctx: usize,
    // TOKENの種類数
    n_vocab: usize,
    // Attention + Feed-Forwardの層の数
    n_layer: usize,
    // バッチサイズ
    n_batch: usize,
}


#[derive(Debug)]
pub struct MultiHeadAttention {
    mask_batch: Tensor,
    // qs, ks, vs: Q, K, V行列をHeadの数だけ準備
    qs: Vec<Linear>,
    ks: Vec<Linear>,
    vs: Vec<Linear>,
    o: Linear,
    // パラメータ
    params: HyperParams,
}

impl MultiHeadAttention {
    pub fn new(params: HyperParams, vb: VarBuilder) -> Self {
        let HyperParams {
            d_model,
            d_head,
            n_head,
            n_ctx,
            n_batch,
            ..
        } = params;

        // softmaxを計算したときに0に近くなればいいので、大きい数字を引く形でmaskを実現する
        // 0をかけるのではなく、softmaxを見込んでかさんで実現するのは、計算が軽いとかモチベーションのはず
        let mut mask = vec![0.0f32; n_ctx * n_ctx];
        for i in 0..n_ctx {
            for j in (i+1)..n_ctx { 
                // 現在の位置 i よりも後ろの位置 j > i がマスク
                // デルは過去のトークンのみを参照できるようになる
                mask[i * n_ctx + j] = -1e9;
            }
        }
        let mask = Tensor::from_vec(mask, (n_ctx, n_ctx), vb.device()).unwrap();
        let mask_batch = mask
            .reshape((1, n_ctx, n_ctx))
            .unwrap()
            .repeat((n_batch, 1, 1))
            .unwrap();

        let mut qs = vec![];
        let mut ks = vec![];
        let mut vs = vec![];
        for i in 0..n_head {
            let q = linear(d_model, d_head, vb.pp(format!("attention_q_{:02}", i))).unwrap();
            let k = linear(d_model, d_head, vb.pp(format!("attention_k_{:02}", i))).unwrap();
            let v = linear(d_model, d_head, vb.pp(format!("attention_v_{:02}", i))).unwrap();
            qs.push(q);
            ks.push(k);
            vs.push(v);
        }

        let o = linear(n_head * d_head, d_model, vb.pp("o")).unwrap();

        Self {
            mask_batch,
            qs,
            ks,
            vs,
            o,
            params,
        }
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let HyperParams {
            d_model,
            d_head,
            n_head,
            n_ctx,
            n_batch,
            ..
        } = self.params;
        // ys: 各HeadについてAttentionのQKV部分を計算したTensor
        let mut ys = vec![];
        for i in 0..n_head {
            let xs_q = self.qs[i].forward(&xs).unwrap();
            let xs_k = self.ks[i].forward(&xs).unwrap();
            let xs_v = self.vs[i].forward(&xs).unwrap();
            // transposeは0-indexedで転置する. この場合は1、2軸を転置で、、各バッチについて行列を転置する
            let xs_1 = xs_q.matmul(&xs_k.transpose(1, 2).unwrap()).unwrap();
            // スケーリングする
            let xs_2 = (&xs_1 / (d_head as f64).sqrt()).unwrap();
            // マスクをかける
            let xs_3 = (&xs_2 + &self.mask_batch).unwrap();
            // softmaxを計算する
            let xs_4 = softmax(&xs_3, D::Minus1).unwrap();
            // [n_batch, n_ctx, n_head]
            let xs_5 = xs_4.matmul(&xs_v).unwrap();
            ys.push(xs_5);
        }
        // 各Headの結果を結合する( Concatenate)
        let xs_6 = Tensor::cat(&ys, 2).unwrap();
        // [n_batch * n_ctx, n_head * d_head]
        let xs_6_re = &xs_6.reshape((n_batch * n_ctx, n_head * d_head)).unwrap();
        // 出力層のLinearをかける [n_batch * n_ctx, d_model]
        let xs_o = self.o.forward(&xs_6_re).unwrap();
        // [n_batch, n_ctx, d_model]
        let xs_o_re = xs_o.reshape((n_batch, n_ctx, d_model)).unwrap();
        Ok(xs_o_re)
    }
}


#[derive(Debug)]
pub struct Layer {
    attention: MultiHeadAttention,
    layer_norm_1: LayerNorm, 
    dense_1: Linear,
    dense_relu: Activation,
    dense_2: Linear,
    layer_norm_2: LayerNorm,
}

impl Layer {
    pub fn new(params: HyperParams, vb: VarBuilder) -> Self {
        let HyperParams {
            d_model,
            ..
        } = params;
        let layer_norm_1 = layer_norm(
            d_model,
            LayerNormConfig::default(),
            vb.pp("layer_norm_attention")
        ).unwrap();
        let attention = MultiHeadAttention::new(params, vb.pp("attention"));
        let layer_norm_2 = layer_norm(
            d_model, 
            LayerNormConfig::default(), 
            vb.pp("layer_norm_feed_forward")
        ).unwrap();
        let dense_1 = linear(d_model, 4 * d_model, vb.pp("dense_1")).unwrap();
        let dense_relu = Activation::Relu;
        let dense_2 = linear(4 * d_model, d_model, vb.pp("dense_2")).unwrap();

        Self {
            layer_norm_1,
            attention,
            layer_norm_2,
            dense_1,
            dense_relu,
            dense_2,
        }
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let xs_1 = self.attention.forward(&xs).unwrap();
        let xs_2 = (&xs_1 + xs).unwrap();
        let xs_3 = self.layer_norm_1.forward(&xs_2).unwrap();
        let xs_4 = self.dense_1.forward(&xs_3).unwrap();
        let xs_5 = self.dense_relu.forward(&xs_4).unwrap();
        let xs_6 = self.dense_2.forward(&xs_5).unwrap();
        let xs_7 = (&xs_6 + &xs_3).unwrap();
        let xs_8 = self.layer_norm_2.forward(&xs_7).unwrap();
        Ok(xs_8)
    }
}


fn positional_encoding_tensor(
    device: &Device,
    d_model: usize,
    n_ctx: usize,
    n_batch: usize,
) -> Tensor {
    let mut pe = vec![0.0f32; d_model * n_ctx];
    for pos in 0..n_ctx {
        for i in 0..d_model / 2 {
            pe[pos * d_model + 2 * i] = (pos as f32 / 10000f32.powf(2.0 * i as f32 / d_model as f32)).sin();
            pe[pos * d_model + 2 * i + 1] = (pos as f32 / 10000f32.powf(2.0 * i as f32 / d_model as f32)).cos();

        }
    }
    let pe = Tensor::from_vec(pe, (1, n_ctx, d_model), device)
        .unwrap()
        .repeat((n_batch, 1))
        .unwrap();
    pe
}

#[derive(Debug)]
pub struct LanguageModel {
    // Token ID -> d_model次元のベクトルに変換
    embedding: Embedding,
    positional_encoding_tensor: Tensor,
    layers: Vec<Layer>,
    output_layer: Linear,
}

impl LanguageModel {
    pub fn new(device: &Device, params: HyperParams, vb: &VarBuilder) -> Self {
        let HyperParams {
            d_model,
            n_ctx,
            n_vocab,
            n_layer,
            n_batch,
            ..
        } = params;
        let embedding = embedding(n_vocab, d_model, vb.pp("embedding")).unwrap();
        let positional_encoding_tensor = positional_encoding_tensor(&device, d_model, n_ctx, n_batch);

        let mut layers = vec![];
        for i in 0..n_layer {
            let layer = Layer::new(params, vb.pp(format!("layer_{:02}", i)));
            layers.push(layer);
        }

        let output_layer = linear(d_model, n_vocab, vb.pp("output_layer")).unwrap();

        Self {
            embedding,
            positional_encoding_tensor,
            layers,
            output_layer,
        }
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        // embedding
        let xs_1 = self.embedding.forward(&xs).unwrap();
        // positional encoding
        let xs_2 = (&xs_1 + &self.positional_encoding_tensor).unwrap();
        // layers
        let mut xs_3 = xs_2.clone();
        for layer in &self.layers {
            xs_3 = layer.forward(&xs_3).unwrap();
        }
        // output layer
        let xs_4 = self.output_layer.forward(&xs_3).unwrap();
        Ok(xs_4)
    }
}

#[derive(Debug)]
pub struct TrainData {
    // [n_batch, n_ctx]
    pub input: Tensor,
    // [n_batch, n_ctx]
    pub expected_output: Tensor,
}

fn create_add_binary_exprs(hyper_params: HyperParams, device: &Device) -> TrainData {
    let HyperParams {
        n_ctx,
        n_vocab,
        n_batch,
        ..
    } = hyper_params;
    assert!(n_vocab == 7);
    // input
    let mut input = vec![0i64; n_batch * n_ctx];
    let mut expected_output = vec![0i64; n_batch * n_ctx];
    let mut rand = thread_rng();
    let max_digits = 12;
    for i in 0..n_batch {
        // 桁数を1~max_digitsでuniformに持ってきて、その桁数の数をuniformに持ってきて値を作る
        // 素直にuniformに持ってくると大きい桁数に偏るため
        let a_digits = rand.gen_range(1..=max_digits);
        let a = rand.gen_range((1 << (a_digits - 1))..(1 << a_digits));
        let b_digits = rand.gen_range(1..=max_digits);
        let b = rand.gen_range((1 << (b_digits - 1))..(1 << b_digits));
        let c = a + b;
        // 数字を書く順番を逆順ではなく普通の順番にしたいときは`.rev()`を消す
        let a_s = format!("{:b}", a).chars().rev().collect::<String>();
        let b_s = format!("{:b}", b).chars().rev().collect::<String>();
        let c_s = format!("{:b}", c).chars().rev().collect::<String>();
        let s = format!("{} + {} = {}.", a_s, b_s, c_s);
        let cs = s.chars().collect::<Vec<_>>();
        assert!(cs.len() < n_ctx);
        for k in 0..n_ctx {
            if k < cs.len() {
                input[i * n_ctx + k] = VOCABS.iter().position(|&c| c == cs[k]).unwrap() as i64;
            } else {
                // <PAD>
                input[i * n_ctx + k] = 0;
            }
        }
        // 期待出力は入力を1ずらして作る
        for k in 0..n_ctx - 1 {
            expected_output[i * n_ctx + k] = input[i * n_ctx + k + 1];
        }
        expected_output[i * n_ctx + (n_ctx - 1)] = 0;
    }
    let input = Tensor::from_vec(input, (n_batch, n_ctx), device).unwrap();
    let expected_output = Tensor::from_vec(expected_output, n_batch * n_ctx, device).unwrap();
    TrainData {
        input,
        expected_output,
    }
}


// ヘルパー関数: トークンIDを文字列に変換
fn tokens_to_string(tokens: &[i64]) -> String {
    tokens.iter()
        .map(|&x| VOCABS.get(x as usize).unwrap_or(&'_'))
        .collect()
}

// データを表示する関数（index_selectを使用せず）
fn print_train_data(
    train_data: &TrainData,
    n_ctx: usize,
    n_batch: usize,
    num_samples: usize,
) {
    // expected_output のテンソルを [n_batch, n_ctx] に再構成
    let expected_output = train_data.expected_output.reshape((n_batch, n_ctx)).unwrap();
    
    // input のテンソルを [n_batch, n_ctx] に再構成
    let input = train_data.input.reshape((n_batch, n_ctx)).unwrap();

    // テンソルを2次元ベクターに変換
    let input_vec = input.to_vec2::<i64>().unwrap();
    let expected_output_vec = expected_output.to_vec2::<i64>().unwrap();

    for i in 0..num_samples.min(n_batch) {
        // 各サンプルの入力シーケンスと期待される出力シーケンスを取得
        let input_seq = &input_vec[i];
        let expected_output_seq = &expected_output_vec[i];

        // トークンIDを文字列に変換
        let input_str = tokens_to_string(input_seq);
        let expected_output_str = tokens_to_string(expected_output_seq);

        println!("Sample {}:", i + 1);
        println!("  Input:           {}", input_str);
        println!("  Expected Output: {}", expected_output_str);
        println!();
    }
}


fn main() {
    // デバイス
    let device = Device::cuda_if_available(0).unwrap();
    println!("device: {:?}", device);
    // パラメータ
    let var_map = VarMap::new();
    let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);

    // ハイパーパラメータ
    let params = HyperParams {
        d_model: 96,
        d_head: 24,
        n_head: 4, 
        n_ctx: 64,
        n_vocab: 7,
        n_layer: 4,
        n_batch: 192,
    };

    const MAX_ITER: usize = 10000;

    let HyperParams {
        n_ctx,
        n_vocab,
        n_batch,
        ..
    } = params;

    // 言語モデル全体
    let model = LanguageModel::new(&device, params, &vb);
    
    // optimizer
    let mut optimizer = AdamW::new(var_map.all_vars(), ParamsAdamW { lr: 1e-3, ..Default::default() }).unwrap();

    // 学習ループ
    // ログファイルを作成
    let mut log_file = File::create("training_log.txt").unwrap();
    use std::io::Write;

    for iter in 0..MAX_ITER {
        // 　学習データ準備
        let TrainData {
            input,
            expected_output,
        } = create_add_binary_exprs(params, &device);

        // データ確認（最初のイテレーションのみ表示）
        if iter == 0 {
            println!("=== Training Data Samples ===");
            print_train_data(&TrainData { input: input.clone(), expected_output: expected_output.clone() }, n_ctx, n_batch, 5);
            println!("==============================\n");
        }

        // forward
        let output = model.forward(&input).unwrap();

        // cross entropy loss
        // Alternative manual cross-entropy loss calculation (no gather)
        let output = output.reshape((n_batch * n_ctx, n_vocab)).unwrap();
        let loss = cross_entropy(&output, &expected_output).unwrap();
        let loss_f32 = loss.to_scalar::<f32>().unwrap();
        let log_str = format!("iter = {}, loss = {}\n", iter, loss_f32);
        log_file.write_all(log_str.as_bytes()).unwrap();
        eprint!("{}", log_str);

        // 時々性能を表示
        if iter % 10 == 5 {
            eprintln!("--------------------------------- iter = {} ---------------------------------", iter);
            let output_softmax = softmax(&output, candle_core::D::Minus1).unwrap();
            let output_softmax_argmax = output_softmax.argmax(candle_core::D::Minus1).unwrap();
            let output_softmax_argmax_vec = output_softmax_argmax.to_vec1::<u32>().unwrap();
            let output_softmax_argmax_vec = output_softmax_argmax_vec
                .iter()
                .map(|&x| x as i64)
                .collect::<Vec<_>>();
            let input_vec = input
                .reshape(n_batch * n_ctx)
                .unwrap()
                .to_vec1::<i64>()
                .unwrap();

            let mut count = 0;
            for i in 0..n_batch {
                // indexの列から文字列を復元したものと、=の後ろを切り出したものを返す関数
                let f = |xs: &[i64]| {
                    let mut s = String::new();
                    for &x in xs {
                        let c = VOCABS[x as usize];
                        s.push(c);
                    }
                    let ts = s.split('=').map(|s| s.to_string()).collect::<Vec<_>>();
                    let ans = ts.get(1).map(|s| s.to_string()).unwrap_or("".to_string());
                    let ans = ans.trim_end_matches('_').to_string();
                    (s, ans)
                };
                let (s, ans_o) = f(&output_softmax_argmax_vec[i * n_ctx..(i + 1) * n_ctx]);
                eprintln!("output = {}", s);
                let (s, ans_i) = f(&input_vec[i * n_ctx..(i + 1) * n_ctx]);
                eprintln!("input  = {}", s);
                eprintln!();
                if ans_o == ans_i {
                    count += 1;
                }
            }
            eprintln!(
                "iter = {}, accuracy: {} / {}, {:.3} %",
                iter,
                count,
                n_batch,
                count as f64 / n_batch as f64 * 100.0
            );
            eprintln!();
        }

        // 勾配計算 
        loss.backward().unwrap();

        // パラメータ更新
        optimizer.backward_step(&loss).unwrap();
    }

    // モデルの保存
    var_map.save("model.bin").unwrap();
}