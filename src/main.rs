use ndarray::{s, Array2, Array3};
use std::f64::consts::PI;
mod plot;
use plot::plot_heatmap;

const GAMMA: f64 = 0.9; // 割引率
const THETA: f64 = 1e-6; // 収束のしきい値
const SIZE: usize = 50; // グリッドサイズ
const T_SIZE: usize = 8; // 角度の離散化数
const MAX_ITER: usize = 1000; // 価値反復の最大回数

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
    let mut values = Array3::from_elem((SIZE, SIZE, T_SIZE), -100.0);
    for t in 0..T_SIZE {
        values[(goal.0, goal.1, t)] = 0.0;
    }

    // アクションの生成
    let actions = generate_actions(T_SIZE);

    // 価値反復法の実行
    for _ in 0..MAX_ITER {
        let mut delta: f64 = 0.0;
        let old_values = values.clone();
        for i in 0..SIZE {
            for j in 0..SIZE {
                for k in 0..T_SIZE {
                    if (i, j) == goal {
                        continue;
                    }
                    let v = values[(i, j, k)];
                    values[(i, j, k)] = compute_value(&old_values, &rewards, &actions, i, j, k);
                    delta = delta.max((v - values[(i, j, k)]).abs());
                }
            }
        }
        if delta < THETA {
            break;
        }
    }

    // 角度0の平面をArray2に変換してプロット
    let values_2d = values.slice(s![.., .., 0]).to_owned();
    plot_heatmap(&values_2d).unwrap();
}

// アクションを生成する関数
fn generate_actions(t_size: usize) -> Vec<Action> {
    let mut actions = Vec::new();

    // 各角度に対する前進と側方移動のアクションを生成
    for i in 0..t_size {
        let angle = i as f64 * 2.0 * PI / t_size as f64;
        actions.push(Action {
            delta_x: angle.cos(),
            delta_y: angle.sin(),
            delta_rot: 0.0,
        });
    }

    // 対角方向のアクションを追加
    let diagonal_moves = vec![(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)];
    for &(dx, dy) in &diagonal_moves {
        actions.push(Action {
            delta_x: dx,
            delta_y: dy,
            delta_rot: 0.0,
        });
    }

    // 回転アクションを生成
    let delta_rotations = vec![PI / 4.0, -PI / 4.0, PI / 2.0];
    for &delta_rot in &delta_rotations {
        actions.push(Action {
            delta_x: 0.0,
            delta_y: 0.0,
            delta_rot,
        });
    }

    actions
}

// 新しい価値を計算する関数
fn compute_value(
    values: &Array3<f64>,
    rewards: &Array2<f64>,
    actions: &Vec<Action>,
    i: usize,
    j: usize,
    k: usize,
) -> f64 {
    let mut max_value = f64::NEG_INFINITY;

    // 各アクションに対して価値を計算
    for action in actions {
        let angle = k as f64 * 2.0 * PI / T_SIZE as f64;
        let ni = (i as isize
            + (action.delta_x * angle.cos() - action.delta_y * angle.sin()).round() as isize)
            as usize;
        let nj = (j as isize
            + (action.delta_x * angle.sin() + action.delta_y * angle.cos()).round() as isize)
            as usize;
        let nk = ((k as isize
            + (action.delta_rot * T_SIZE as f64 / (2.0 * PI)).round() as isize
            + T_SIZE as isize)
            % T_SIZE as isize) as usize;

        // 範囲外チェック
        if ni >= SIZE || nj >= SIZE {
            continue;
        }

        // 価値を計算
        let value = rewards[(i, j)] + GAMMA * values[(ni, nj, nk)];
        if value > max_value {
            max_value = value;
        }
    }

    max_value
}
