#include "plotdialog.h"
//#include <QRandomGenerator>
#include <QtCharts/QChartView>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFrame>
#include <QLabel>
#include <QPushButton>
#include <QPainter>
#include <QColor>
#include <QPen>
#include <QtGlobal>

// 构造函数类名改为 PlotDialog
PlotDialog::PlotDialog(QWidget *parent)
    : QDialog(parent)
    , m_splineSeries(new QSplineSeries(this)) // 样条曲线（天然平滑）
    , m_splineSeries2(new QSplineSeries(this)) // 样条曲线（天然平滑）
       , m_scatterSeries(new QScatterSeries(this))
       , m_chart(new QChart())
       , m_axisX(new QValueAxis(this))
       , m_axisY(new QValueAxis(this))
   {
        this->rendering_data = NULL;

       setWindowTitle("Neural Manifold");
       setMinimumSize(1600, 600);
       setModal(true);
       setAttribute(Qt::WA_DeleteOnClose);

       QVBoxLayout *mainLayout = new QVBoxLayout(this);
       mainLayout->setContentsMargins(20, 20, 20, 20);
       mainLayout->setSpacing(15);

       QLabel *titleLabel = new QLabel("Neural Manifold computational results", this);
       titleLabel->setAlignment(Qt::AlignCenter);
       QFont titleFont = titleLabel->font();
       titleFont.setPointSize(16);
       titleFont.setBold(true);
       titleLabel->setFont(titleFont);
       mainLayout->addWidget(titleLabel);

       QFrame *chartFrame = new QFrame(this);
       chartFrame->setStyleSheet(R"(
           QFrame {
               background-color: white;
               border-radius: 8px;
               border: 1px solid #e0e0e0;
           }
       )");
       QVBoxLayout *chartLayout = new QVBoxLayout(chartFrame);
       chartLayout->setContentsMargins(10, 10, 10, 10);

       // ========== 1. 样条曲线配置（天然平滑，Qt5.12+ 原生支持） ==========
       m_splineSeries->setName("real data");
       // 可选：优化线条样式，让平滑效果更明显
       QPen splinePen(Qt::blue);
       splinePen.setWidthF(1.5);       // 浮点宽度，更细腻
       splinePen.setCapStyle(Qt::RoundCap); // 端点圆角
       splinePen.setJoinStyle(Qt::RoundJoin); // 拐角圆角
       m_splineSeries->setPen(splinePen);

       // 可选：优化线条样式，让平滑效果更明显
       QPen splinePen2(Qt::red);
       splinePen2.setWidthF(2.5);       // 浮点宽度，更细腻
       splinePen2.setCapStyle(Qt::RoundCap); // 端点圆角
       splinePen2.setJoinStyle(Qt::RoundJoin); // 拐角圆角
       m_splineSeries2->setPen(splinePen2);

       // ========== 2. 散点系列配置（仅100-106红色点） ==========
       m_scatterSeries->setName("predicted data");
       m_scatterSeries->setMarkerSize(8);
       m_scatterSeries->setColor(Qt::red);
       m_scatterSeries->setBorderColor(Qt::darkRed);
       //m_scatterSeries->setBorderWidth(1);

       // ========== 3. 添加系列到图表 ==========
       m_chart->addSeries(m_splineSeries);
       m_chart->addSeries(m_splineSeries2);
       m_chart->addSeries(m_scatterSeries);
       m_chart->setAnimationOptions(QChart::SeriesAnimations);
       m_chart->setBackgroundVisible(false);

       // ========== 4. 坐标轴配置 ==========
       m_axisX->setLabelFormat("%d");
       m_axisX->setTickCount(10);
       m_axisX->setMinorTickCount(1);
       m_axisX->setGridLineColor(QColor(0xe0e0e0));

       m_axisY->setRange(-100, 100);
       m_axisY->setLabelFormat("%.2f");
       m_axisY->setTitleText("Value");
       m_axisY->setTickCount(11);
       m_axisY->setMinorTickCount(1);
       m_axisY->setGridLineColor(QColor(0xe0e0e0));

       // ========== 5. 绑定坐标轴 ==========
       m_chart->addAxis(m_axisX, Qt::AlignBottom);
       m_chart->addAxis(m_axisY, Qt::AlignLeft);
       m_splineSeries->attachAxis(m_axisX);
       m_splineSeries->attachAxis(m_axisY);
       m_splineSeries2->attachAxis(m_axisX);
       m_splineSeries2->attachAxis(m_axisY);
       m_scatterSeries->attachAxis(m_axisX);
       m_scatterSeries->attachAxis(m_axisY);

       // ========== 6. 图表视图（抗锯齿必加） ==========
       QChartView *chartView = new QChartView(m_chart, this);
       chartView->setRenderHint(QPainter::Antialiasing); // 消除锯齿，平滑效果更优
       chartView->setMinimumHeight(400);
       chartLayout->addWidget(chartView);
       mainLayout->addWidget(chartFrame);

       // ========== 7. 按钮区域 ==========
       QHBoxLayout *btnLayout = new QHBoxLayout();
       QPushButton *closeBtn = new QPushButton("Close", this);

       btnLayout->addStretch();
       btnLayout->addWidget(closeBtn);
       btnLayout->addStretch();
       mainLayout->addLayout(btnLayout);

       // ========== 8. 信号槽 ==========
       connect(closeBtn, &QPushButton::clicked, this, &QDialog::close);

       // ========== 9. 样式表 ==========
       setStyleSheet(R"(
           QDialog {
               background-color: #f5f5f5;
           }
           QPushButton {
               padding: 8px 20px;
               font-size: 14px;
               border: none;
               border-radius: 4px;
               background-color: #2196F3;
               color: white;
           }
           QPushButton:hover {
               background-color: #1976D2;
           }
           QPushButton:pressed {
               background-color: #0D47A1;
           }
           QLabel {
               color: #333333;
           }
       )");
}

void PlotDialog::setData(float *data,unsigned int length)
{
    if(NULL != this->rendering_data){
        this->rendering_data = (float*)realloc(this->rendering_data,length*sizeof(float));
    }else{
        this->rendering_data = (float*)malloc(length*sizeof(float));
    }

    memcpy(this->rendering_data,data,length*sizeof(float));

    float max = -9999999;
    float min = 9999999;
    for(unsigned int i=0;i<length;i++)
    {
        if(data[i] > max){
            max = data[i];
        }

        if(data[i] < min)
        {
            min = data[i];
        }
    }

    float median = (max-min)/2 + min;
    m_axisY->setRange((min - (median-min)*0.5),max + (max-median)*0.5);

    m_splineSeries->clear();
    for (unsigned int x = 0; x < length; ++x) {
        m_splineSeries->append(x, data[x]);
    }
}

// 私有函数类名改为 PlotDialog
//void PlotDialog::generateSplineData()
//{
 //   static unsigned int off_set=0;

//    if(NULL == this->rendering_data){return;}
//    off_set++;
//    m_splineSeries->clear();
//    for (int x = 0; x < 200; ++x) {
//        qreal y = this->rendering_data[x+off_set];//rand()%1000/10.0;//QRandomGenerator::global()->bounded(100.0);
//        m_splineSeries->append(x, y);
//    }
//}

// 新增：生成仅100-106横坐标的数据
//void PlotDialog::generateScatterData()
//{
//    // 仅在x=100到x=106之间生成数据（共7个点）
//    for (int x = 100; x <= 106; ++x) {
//        // 随机生成y值（0-100），也可手动指定固定值（如y=50）
//        qreal y = rand()%1000/10.0;
//        // 手动指定值示例：y = 50.0;
//        m_scatterSeries->append(x, y);
//    }
//}

void PlotDialog::refreshData(unsigned int off_set,float *predicted_data,unsigned int length)
{
    m_axisX->setRange(off_set-200, off_set+10);

    //m_splineSeries2->clear();
    m_scatterSeries->clear();
    for(unsigned int i=0;i<length;i++)
    {
        m_splineSeries2->append(off_set+6+i, predicted_data[i]);
        m_scatterSeries->append(off_set+6+i, predicted_data[i]);
    }
}
