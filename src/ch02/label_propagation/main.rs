use ndarray::{Array2, Array1, Axis, array};
use ndarray_linalg::solve::Inverse;
use std::collections::HashSet;

 
// ラベル伝播法
// 教師あり頂点のラベルを教師なし頂点に伝播する
// 
// # Arguments
// * `w`: 隣接行列
// * `v_l`: 教師あり頂点のインデックス
// * `y_l`: 教師あり頂点のラベル
// 
// # Returns
// * 教師なし頂点のラベル
fn label_propagation(
    w: &Array2<f64>,
    v_l: &HashSet<usize>,
    y_l: &Array1<f64>,
) -> Array1<f64> {
    let n = w.shape()[0];
    let mut d = Array2::<f64>::zeros((n, n));

    // 隣接行列の対角成分に次数を設定
    for i in 0..n {
        let degree = w.row(i).sum();
        d[(i, i)] = degree;
    }

    // 教師あり頂点と教師なし頂点のインデックスを取得
    let v_u: HashSet<usize> = (0..n)
        .filter(|i| !v_l.contains(i))
        .collect();

    // 行列を教師あり部分と教師なし部分に分割
    let w_uu = w
        .select(Axis(0), &v_u.iter().cloned().collect::<Vec<_>>())
        .select(Axis(1), &v_u.iter().cloned().collect::<Vec<_>>());
    let w_ul = w
        .select(Axis(0), &v_u.iter().cloned().collect::<Vec<_>>())
        .select(Axis(1), &v_l.iter().cloned().collect::<Vec<_>>());

    // 逆行列の計算
    let d_uu = d
        .select(Axis(0), &v_u.iter().cloned().collect::<Vec<_>>())
        .select(Axis(1), &v_u.iter().cloned().collect::<Vec<_>>());

    let inv_d_uu_w_uu = (d_uu - w_uu).inv().unwrap();

    // 教師ありラベルを行列に変換
    let f_l = Array2::from_shape_vec((v_l.len(), 1), y_l.to_vec()).unwrap();

    // 教師なし頂点のラベルを計算
    let f_u = inv_d_uu_w_uu.dot(&w_ul).dot(&f_l);

    // ラベルの予測値を0.5で閾値処理
    f_u.mapv(|x| if x >= 0.5 { 1.0f64 } else { 0.0f64 }).into_shape(v_u.len()).unwrap()
}

fn main() {
    // 重み付き隣接行列の例
    let w = array![
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
    ];

    // 教師あり頂点のインデックス
    let v_l : HashSet<usize> = [0, 3].iter().cloned().collect();

    // 教師あり頂点のラベル
    let y_l = array![1.0, 0.0];

    // ラベル伝播法の実行
    let f_u = label_propagation(&w, &v_l, &y_l);

    // 結果の表示
    println!("{:?}", f_u);
}
