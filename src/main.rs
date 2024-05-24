use ndarray::{s, Array2, Array3};
use rand::Rng;
use std::f64::consts::PI;
use std::time::Instant;
mod plot;
use plot::plot_heatmap;

const GAMMA: f64 = 1.0; // 割引率
const DELTA_LIMIT: f64 = 1e-6; // 収束のしきい値
const SIZE: usize = 50; // グリッドサイズ
const THETA_SIZE: usize = 36; // 角度の離散化数
const MAX_ITER: usize = 1000; // 価値反復の最大回数
const NUM_SAMPLES: usize = 10; // ランダムサンプリングの回数

// アクションを表す構造体
#[derive(Clone)]
struct Action {
    delta_x: f64,   // x方向の移動距離
    delta_y: f64,   // y方向の移動距離
    delta_rot: f64, // 回転角度
}

fn main() {
    // 報酬を初期化。初期値は全て-1.0
    let mut rewards = Array2::from_elem((SIZE, SIZE), -1.0);

    // ゴールの位置を設定
    let goal = (SIZE - 11, SIZE - 11);
    rewards[goal] = 0.0;

    // 初期価値を-100に設定
    let mut values = Array3::from_elem((SIZE, SIZE, THETA_SIZE), -100.0);
    for t in 0..THETA_SIZE {
        values[(goal.0, goal.1, t)] = 0.0;
    }

    // アクションの生成
    let actions = generate_actions();

    // 実行時間の計測開始
    let start = Instant::now();

    // 価値反復法の実行
    for _ in 0..MAX_ITER {
        let mut delta: f64 = 0.0;
        let old_values = values.clone();
        for i in 0..SIZE {
            for j in 0..SIZE {
                for theta in 0..THETA_SIZE {
                    if (i, j) == goal {
                        continue;
                    }
                    let v = values[(i, j, theta)];
                    values[(i, j, theta)] =
                        compute_value(&old_values, &rewards, &actions, i, j, theta);
                    delta = delta.max((v - values[(i, j, theta)]).abs());
                }
            }
        }
        if delta < DELTA_LIMIT {
            break;
        }
    }

    // 実行時間の計測終了
    let duration = start.elapsed();

    // 実行時間を表示
    println!("実行時間: {:?}", duration);

    // 角度0の平面をArray2に変換してプロット
    let values_2d = values.slice(s![.., .., 0]).to_owned();
    plot_heatmap(&values_2d).unwrap();
}

// アクションを生成する関数
fn generate_actions() -> Vec<Action> {
    let mut actions = Vec::new();

    // 右回転
    actions.push(Action {
        delta_x: 0.0,
        delta_y: 0.0,
        delta_rot: -2.0,
    });

    // 前進
    actions.push(Action {
        delta_x: 1.0,
        delta_y: 0.0,
        delta_rot: 0.0,
    });

    // 左回転
    actions.push(Action {
        delta_x: 0.0,
        delta_y: 0.0,
        delta_rot: 2.0,
    });

    actions
}

// 新しい価値を計算する関数
fn compute_value(
    values: &Array3<f64>,
    rewards: &Array2<f64>,
    actions: &Vec<Action>,
    i: usize,
    j: usize,
    theta: usize,
) -> f64 {
    let mut rng = rand::thread_rng();
    let mut total_value = 0.0;
    let mut valid_samples = 0;

    // ランダムにNUM_SAMPLES回サンプリング
    for _ in 0..NUM_SAMPLES {
        // ランダムにアクションを選択
        let action_index = rng.gen_range(0..actions.len());
        let action = &actions[action_index];
        let angle = theta as f64 * 2.0 * PI / THETA_SIZE as f64;

        // 移動後の状態を計算
        let ni = (i as isize
            + (action.delta_x * angle.cos() - action.delta_y * angle.sin()).round() as isize)
            as usize;
        let nj = (j as isize
            + (action.delta_x * angle.sin() + action.delta_y * angle.cos()).round() as isize)
            as usize;
        let ntheta = ((theta as isize
            + (action.delta_rot * THETA_SIZE as f64 / (2.0 * PI)).round() as isize
            + THETA_SIZE as isize)
            % THETA_SIZE as isize) as usize;

        // グリッドの範囲内に収める
        let ni = ni.min(SIZE - 1).max(0);
        let nj = nj.min(SIZE - 1).max(0);

        // 遷移後の状態における報酬と価値の和を計算
        total_value += rewards[(ni, nj)] + GAMMA * values[(ni, nj, ntheta)];

        valid_samples += 1;
    }

    // 遷移前の状態の価値を遷移後の報酬と価値の期待値と等しくする
    total_value / valid_samples as f64
}
