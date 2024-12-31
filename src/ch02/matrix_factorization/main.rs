use ndarray::{array, Array1, Array2, Axis};
use ndarray_linalg::Eig;


// 隣接行列の行列分解
// 
// # Arguments
// * `w`: 重み付き隣接行列 \bm{W} \in \mathbb{R}^{n \times n}
// * `d`: 埋め込みの次元
//
// # Returns
// * `z`: 頂点埋め込み \bm{Z} \in \mathbb{R}^{n \times d}
fn matrix_factorization(
    w: &Array2<f64>,
    d: usize,
) -> Array2<f64> {
    // ステップ 1: 対角行列 D を計算
    let n = w.nrows();
    let mut d_matrix = Array2::<f64>::zeros((n, n));

    for i in 0..n {
        let row_sum: f64 = w.row(i).sum();
        d_matrix[[i, i]] = row_sum;
    }

    // ステップ 2: 固有値分解
    let (eigenvalues, eigenvectors) = (d_matrix + w).eig().unwrap();

    // ステップ 3: 固有値と固有ベクトルをソート（降順）
    let mut eig_pairs: Vec<(f64, Array1<f64>)> = eigenvalues
        .iter()
        .zip(eigenvectors.axis_iter(Axis(1)))
        .map(|(e, v)| (e.re, v.map(|x| x.re).to_owned()))
        .collect();
    eig_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    // ステップ 4: 上位 d 次元を選択して Z を計算
    let mut z = Array2::<f64>::zeros((n, d));

    for i in 0..d {
        let (lambda, v) = &eig_pairs[i];
        let scaled_v = v * lambda.sqrt();
        z.column_mut(i).assign(&scaled_v);
    }

    z

}

fn main() {
    // 重み付き隣接行列の例
    let w = array![
        [0.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
    ];
    let d = 2; // 埋め込み次元

    let z = matrix_factorization(&w, d);
    println!("{:?}", z);
}