use ndarray::{Array2, Array3};

pub fn initialize_goal_values(size: usize, theta_size: usize, values: &mut Array3<f64>) {
    let goal = (size - 11, size - 11);
    for t in 0..theta_size {
        values[(goal.0, goal.1, t)] = 0.0;
    }
}

pub fn initialize_arrays(size: usize, theta_size: usize) -> (Array2<f64>, Array3<f64>) {
    let rewards = Array2::from_elem((size, size), -1.0);
    let values = Array3::from_elem((size, size, theta_size), -100.0);
    (rewards, values)
}

pub fn set_goal(size: usize, rewards: &mut Array2<f64>) {
    let goal = (size - 11, size - 11);
    rewards[goal] = 0.0;
}

pub fn set_boundaries(size: usize, rewards: &mut Array2<f64>) {
    for i in 0..size {
        rewards[(i, 0)] = -5.0;
        rewards[(i, size - 1)] = -100.0;
        rewards[(0, i)] = -5.0;
        rewards[(size - 1, i)] = -100.0;
    }
}

pub fn set_puddle(size: usize, rewards: &mut Array2<f64>) {
    let obstacle_end = 2 * size / 4;
    let puddle_start = obstacle_end;
    let puddle_end = obstacle_end + (obstacle_end - (size / 4)) / 2;
    for i in puddle_start..puddle_end {
        for j in puddle_start..puddle_end {
            if i < size && j < size {
                rewards[(i, j)] = -10.0;
            }
        }
    }
}

pub fn generate_actions() -> Vec<(isize, isize, isize)> {
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

pub fn set_obstacles(size: usize, rewards: &mut Array2<f64>) {
    let obstacle_start = size / 4;
    let obstacle_end = 2 * size / 4;
    for i in obstacle_start..obstacle_end {
        for j in obstacle_start..obstacle_end {
            rewards[(i, j)] = -20.0;
        }
    }
}

pub fn calculate_value(
    i: usize,
    j: usize,
    theta: usize,
    gamma: f64,
    size: usize,
    theta_size: usize,
    action: (isize, isize, isize),
    old_values: &Array3<f64>,
    rewards: &Array2<f64>,
) -> f64 {
    let (next_i, next_j, next_theta) = next_state(i, j, size, theta_size, theta, action);
    let reward = rewards[(next_i, next_j)];
    let move_cost = if action.0 != 0 && action.1 != 0 {
        (2.0 as f64).sqrt()
    } else {
        1.0
    };
    reward + gamma * old_values[(next_i, next_j, next_theta)] - move_cost
}

pub fn next_state(
    i: usize,
    j: usize,
    size: usize,
    theta_size: usize,
    theta: usize,
    action: (isize, isize, isize),
) -> (usize, usize, usize) {
    let ni = ((i as isize + action.0).clamp(0, (size - 1) as isize)) as usize;
    let nj = ((j as isize + action.1).clamp(0, (size - 1) as isize)) as usize;
    let ntheta = ((theta as isize + action.2).rem_euclid(theta_size as isize)) as usize;
    (ni, nj, ntheta)
}
