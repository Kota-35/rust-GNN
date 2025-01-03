# Docker Official の Rust イメージを使用
# ビルド用のイメージを builderと名付ける
FROM rust:1.79.0 as builder

# /rust-gnn でビルドを行うことにする
WORKDIR /rust-gnn

# 1. Cargo.lock (と Cargo.toml) をコピー
COPY Cargo.toml Cargo.lock ./

# 2. クレートをダウンロードだけする。コンパイルはしない。
RUN cargo fetch

# 3. プロジェクトをコピー
COPY . .

# 4. ビルド
RUN cargo build --release

# 先ほどビルドした生成物のうち、アプリケーションのもののみを削除する
RUN rm -f target/release/deps/label_propagation*
RUN rm -f target/release/deps/matrix_factorization*
RUN rm -f target/release/deps/layers*

# 改めてアプリケーションをビルドする
RUN cargo build --release

# 新しくリリース用のイメージを用意する
# リリース用のイメージには  ubuntu を使用する
FROM ubuntu:22.04

# builder イメージからバイナリをコピーして /usr/local/bin にインストールする
COPY --from=builder /rust-gnn/target/release/label_propagation /usr/local/bin/label_propagation
COPY --from=builder /rust-gnn/target/release/matrix_factorization /usr/local/bin/matrix_factorization
COPY --from=builder /rust-gnn/target/release/layers /usr/local/bin/layers

# コンテナ起動時にlayersを実行します
CMD ["/usr/local/bin/layers"]
