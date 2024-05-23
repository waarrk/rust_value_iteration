use ndarray::Array2;
use plotters::prelude::*;

// ヒートマップをプロットする関数
pub fn plot_heatmap(values: &Array2<f64>) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("values_heatmap.png", (1000, 900)).into_drawing_area();
    root.fill(&WHITE)?;

    // 最小値と最大値を求める
    let (min, max) = values
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &val| {
            (min.min(val), max.max(val))
        });

    let (left, right) = root.split_horizontally(850);

    let mut chart = ChartBuilder::on(&left)
        .margin(20)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..values.shape()[1] as i32, 0..values.shape()[0] as i32)?;

    chart
        .configure_mesh()
        .disable_mesh()
        .x_labels(values.shape()[1])
        .y_labels(values.shape()[0])
        .draw()?;

    // ヒートマップを描画
    for i in 0..values.shape()[0] {
        for j in 0..values.shape()[1] {
            let value = values[(i, j)];
            let norm_value = (value - min) / (max - min); // 正規化
            let color = HSLColor(240.0 / 360.0 - norm_value * 240.0 / 360.0, 1.0, 0.5);
            chart.draw_series(std::iter::once(Rectangle::new(
                [
                    (j as i32, values.shape()[0] as i32 - i as i32 - 1),
                    ((j + 1) as i32, values.shape()[0] as i32 - i as i32),
                ],
                color.filled().stroke_width(1),
            )))?;
        }
    }

    // カラーバーを描画
    let mut chart_2 = ChartBuilder::on(&right)
        .margin(20)
        .set_label_area_size(LabelAreaPosition::Right, 30)
        .x_label_area_size(30)
        .build_cartesian_2d(0..1, min..max)?;

    chart_2
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .disable_x_axis()
        .draw()?;

    chart_2.draw_series((0..1000).map(|i| {
        let value = min + (max - min) * i as f64 / 1000.0;
        let next_value = min + (max - min) * (i + 1) as f64 / 1000.0;
        let norm_value = i as f64 / 1000.0; // 正規化
        let color = HSLColor(240.0 / 360.0 - norm_value * 240.0 / 360.0, 1.0, 0.5);
        Rectangle::new([(0, value), (1, next_value)], color.filled())
    }))?;
    root.present()?;
    Ok(())
}
