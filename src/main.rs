use ndarray::{s, Array2, Array3};
use std::time::Instant;
mod plot;
use plot::plot_heatmap;

const GAMMA: f64 = 1.0; // 割引率
const DELTA_LIMIT: f64 = 1.0 + 1e-6; // 収束判定の閾値
const SIZE: usize = 100; // グリッドサイズ
const THETA_SIZE: usize = 36; // 角度の離散化数
const MAX_ITER: usize = 1000; // 価値反復の最大回数

fn main() {
    let (mut rewards, mut values) = initialize_arrays();
    set_goal(&mut rewards);
    set_boundaries(&mut rewards);
    set_obstacles(&mut rewards);
    set_puddle(&mut rewards);

    initialize_goal_values(&mut values);

    let actions = generate_actions();

    // 計測開始
    let start = Instant::now();

    // 価値反復アルゴリズムの実行
    value_iteration(&mut values, &rewards, &actions);

    // 計測終了
    let duration = start.elapsed();
    println!("実行時間: {:?}", duration);

    // 価値関数の描画
    let values_2d = values.slice(s![.., .., 0]).to_owned();
    plot_heatmap(&values_2d).unwrap();
}

fn initialize_arrays() -> (Array2<f64>, Array3<f64>) {
    let rewards = Array2::from_elem((SIZE, SIZE), -1.0);
    let values = Array3::from_elem((SIZE, SIZE, THETA_SIZE), -100.0);
    (rewards, values)
}

fn set_goal(rewards: &mut Array2<f64>) {
    let goal = (SIZE - 11, SIZE - 11);
    rewards[goal] = 0.0;
}

fn set_boundaries(rewards: &mut Array2<f64>) {
    for i in 0..SIZE {
        rewards[(i, 0)] = -5.0;
        rewards[(i, SIZE - 1)] = -100.0;
        rewards[(0, i)] = -5.0;
        rewards[(SIZE - 1, i)] = -100.0;
    }
}

fn set_obstacles(rewards: &mut Array2<f64>) {
    let obstacle_start = SIZE / 4;
    let obstacle_end = 2 * SIZE / 4;
    for i in obstacle_start..obstacle_end {
        for j in obstacle_start..obstacle_end {
            rewards[(i, j)] = -20.0;
        }
    }
}

fn set_puddle(rewards: &mut Array2<f64>) {
    let obstacle_end = 2 * SIZE / 4;
    let puddle_start = obstacle_end;
    let puddle_end = obstacle_end + (obstacle_end - (SIZE / 4)) / 2;
    for i in puddle_start..puddle_end {
        for j in puddle_start..puddle_end {
            if i < SIZE && j < SIZE {
                rewards[(i, j)] = -10.0;
            }
        }
    }
}

fn initialize_goal_values(values: &mut Array3<f64>) {
    let goal = (SIZE - 11, SIZE - 11);
    for t in 0..THETA_SIZE {
        values[(goal.0, goal.1, t)] = 0.0;
    }
}

fn generate_actions() -> Vec<(isize, isize, isize)> {
    vec![
        (0, 1, 0),   // 右
        (1, 0, 0),   // 下
        (0, -1, 0),  // 左
        (-1, 0, 0),  // 上
        (1, 1, 0),   // 右下
        (-1, 1, 0),  // 左下
        (1, -1, 0),  // 右上
        (-1, -1, 0), // 左上
        (0, 0, 1),   // 時計回り
        (0, 0, -1),  // 反時計回り
    ]
}

fn value_iteration(
    values: &mut Array3<f64>,
    rewards: &Array2<f64>,
    actions: &[(isize, isize, isize)],
) {
    for iter in 0..MAX_ITER {
        let mut delta: f64 = 0.0;
        let old_values = values.clone();

        for i in 0..SIZE {
            for j in 0..SIZE {
                for theta in 0..THETA_SIZE {
                    // 全ての行動に対して最大の価値を計算
                    let max_value = actions
                        .iter()
                        .map(|action| calculate_value(i, j, theta, *action, &old_values, rewards))
                        .fold(f64::NEG_INFINITY, f64::max);

                    // 価値関数の更新
                    delta = delta.max((max_value - values[(i, j, theta)]).abs());
                    values[(i, j, theta)] = max_value;
                }
            }
        }

        println!("{}回目: delta = {}", iter, delta);

        if delta < DELTA_LIMIT {
            println!("収束しました: {} 回で終了", iter);
            break;
        }
    }
}

fn calculate_value(
    i: usize,
    j: usize,
    theta: usize,
    action: (isize, isize, isize),
    old_values: &Array3<f64>,
    rewards: &Array2<f64>,
) -> f64 {
    let (next_i, next_j, next_theta) = next_state(i, j, theta, action);
    let reward = rewards[(next_i, next_j)];
    let move_cost = if action.0 != 0 && action.1 != 0 {
        (2.0 as f64).sqrt()
    } else {
        1.0
    };
    reward + GAMMA * old_values[(next_i, next_j, next_theta)] - move_cost
}

fn next_state(
    i: usize,
    j: usize,
    theta: usize,
    action: (isize, isize, isize),
) -> (usize, usize, usize) {
    let ni = ((i as isize + action.0).clamp(0, (SIZE - 1) as isize)) as usize;
    let nj = ((j as isize + action.1).clamp(0, (SIZE - 1) as isize)) as usize;
    let ntheta = ((theta as isize + action.2).rem_euclid(THETA_SIZE as isize)) as usize;
    (ni, nj, ntheta)
}
