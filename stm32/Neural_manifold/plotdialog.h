#ifndef PLOTDIALOG_H
#define PLOTDIALOG_H


#include <QDialog>
#include <QtCharts/QSplineSeries>  // 替换：样条曲线（平滑）
#include <QtCharts/QScatterSeries> // 散点系列（红色点）
#include <QtCharts/QChart>
#include <QtCharts/QValueAxis>

//class QVBoxLayout;
//class QHBoxLayout;
//class QFrame;
//class QLabel;
//class QPushButton;
//class QChartView;

QT_CHARTS_USE_NAMESPACE

// 类名改为 PlotDialog
class PlotDialog : public QDialog
{
    Q_OBJECT

public:
    explicit PlotDialog(QWidget *parent = nullptr);
    ~PlotDialog() override = default;

    void setData(float *data,unsigned int length);
    void refreshData(unsigned int off_set,float *predicted_data,unsigned int length);

private:
    QSplineSeries *m_splineSeries; // 平滑样条曲线（替代QLineSeries）
    QSplineSeries *m_splineSeries2;
    QScatterSeries *m_scatterSeries; // 红色散点
    QChart *m_chart;
    QValueAxis *m_axisX;
    QValueAxis *m_axisY;

    float *rendering_data;
};

#endif // PLOTDIALOG_H
