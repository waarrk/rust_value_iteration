use ndarray::{s, Array2, Array3};
use std::time::Instant;
mod plot;
use plot::plot_heatmap;
mod common;
use common::{
    calculate_value, generate_actions, initialize_arrays, initialize_goal_values, set_boundaries,
    set_goal, set_obstacles, set_puddle,
};

const GAMMA: f64 = 1.0; // 割引率
const DELTA_LIMIT: f64 = 1.0 + 1e-6; // 収束判定の閾値
const SIZE: usize = 100; // グリッドサイズ
const THETA_SIZE: usize = 36; // 角度の離散化数
const MAX_ITER: usize = 1000; // 価値反復の最大回数

fn main() {
    let (mut rewards, mut values) = initialize_arrays(SIZE, THETA_SIZE);
    set_goal(SIZE, &mut rewards);
    set_boundaries(SIZE, &mut rewards);
    set_obstacles(SIZE, &mut rewards);
    set_puddle(SIZE, &mut rewards);

    initialize_goal_values(SIZE, THETA_SIZE, &mut values);

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
                        .map(|action| {
                            calculate_value(
                                i,
                                j,
                                theta,
                                GAMMA,
                                SIZE,
                                THETA_SIZE,
                                *action,
                                &old_values,
                                rewards,
                            )
                        })
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
