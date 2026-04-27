#ifndef MAINWIDGET_H
#define MAINWIDGET_H

#include "channelselectlayout.h"
#include "controldialog.h"
#include "hardwarethread.h"
#include "plotdialog.h"

#include <QProgressDialog>
#include <QWidget>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWidget; }
QT_END_NAMESPACE

#define MAX_ROW 400
#define MAX_COL 20

class MainWidget : public QWidget
{
    Q_OBJECT

public:
    MainWidget(QWidget *parent = nullptr);
    ~MainWidget();

    void setUTBoardOKState(bool active);
    void setAutoCycingState(bool active);

private slots:
    void updateDataPosition(int row,int col);
    void updateProgressBar(float progress);
    void updateResult(unsigned int off_set,float *result,unsigned int length);

    void updateCycleState(bool enable);
    void updatePulseState(bool enable);
    void popupMessage(bool err,QString msg);
    void createPolylineDialog(void);

    void on_pushButton_pulse_clicked();
    void on_pushButton_exit_clicked();

    void showControlDialog(void);

    void on_hardwareControlFinish();

    void on_pushButton_load_data_clicked();

    void on_pushButton_open_device_clicked();



private:
    ChannelSelectLayout *channelSelectLayout;
    Ui::MainWidget *ui;

    HardwareThread *hardware_thread;
    ControlDialog *control_dialog;
    QProgressDialog *data_transmission_progress_dialog;
    PlotDialog *plot_dialog;

    bool ut_board_ok;

    void windowsHide(void);

    float  data_matrix[MAX_ROW][MAX_COL];
    int actual_rows;
};
#endif // MAINWIDGET_H
