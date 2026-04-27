#include "controldialog.h"
#include "hardwarethread.h"
#include "loaddata.h"
#include "main_widget.h"
#include "ui_mainwidget.h"

#include <QDebug>
#include <QString>
#include <QMessageBox>
#include <buttonstyle.h>
#include <scrollareastyle.h>
#include <channelselectlayout.h>

#include <QFileDialog>
#include <QProgressDialog>
#include <configparammanager.h>

MainWidget::MainWidget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::MainWidget)
{
    ui->setupUi(this);

    actual_rows = 0;

    this->setWindowTitle("Neural Manifold");

    showMaximized();

    //QMessageBox::information(NULL,"Tips","System start...");
    hardware_thread = new HardwareThread();
    connect(hardware_thread,&HardwareThread::updateRunState,this,&MainWidget::updateCycleState);
    connect(hardware_thread,&HardwareThread::updateConnectState,this,&MainWidget::updatePulseState);
    connect(hardware_thread,&HardwareThread::updateProgress,this,&MainWidget::updateProgressBar);
    connect(hardware_thread,&HardwareThread::updateResult,this,&MainWidget::updateResult);
    connect(hardware_thread,&HardwareThread::startExecution,this,&MainWidget::createPolylineDialog);
    connect(hardware_thread,&HardwareThread::popupMessage,this,&MainWidget::popupMessage);
    connect(hardware_thread,&HardwareThread::finished,this,&MainWidget::on_hardwareControlFinish);

    ui->ChannalSelectScrollArea->setStyleSheet(ScrollAreaStyle::getChannleSelectScrollAreaStyle());    
    channelSelectLayout = new ChannelSelectLayout();
    connect(channelSelectLayout,&ChannelSelectLayout::updateDataPosition,this,&MainWidget::updateDataPosition);
    ui->ChannelSelectWidget->setLayout(channelSelectLayout);
    setWindowFlags(Qt::FramelessWindowHint | Qt::WindowMinMaxButtonsHint);  //去掉窗口边框

    ui->pushButton_load_data->setStyleSheet(ButtonStyle::btLoadDataStyle());
    ui->pushButton_load_data->setEnabled(true);
    ui->pushButton_open_device->setStyleSheet(ButtonStyle::btOpenDeviceStyle());
    ui->pushButton_open_device->setEnabled(false);

    connect(ui->pushButton_maximize,&QPushButton::clicked,this,&MainWidget::showControlDialog);

    control_dialog = new ControlDialog(this);
    connect(control_dialog->getPulseButton(),&QPushButton::clicked,this,&MainWidget::on_pushButton_pulse_clicked);
    connect(control_dialog->getExitButton(),&QPushButton::clicked,this,&MainWidget::on_pushButton_exit_clicked);
    connect(control_dialog,&ControlDialog::finished,this,&MainWidget::showMaximized);
    connect(hardware_thread,&HardwareThread::updateRunState,control_dialog,&ControlDialog::updateCycleState);
    connect(hardware_thread,&HardwareThread::updateConnectState,control_dialog,&ControlDialog::updatePulseState);

    plot_dialog = new PlotDialog(this);
}

MainWidget::~MainWidget()
{
    delete ui;
}

void MainWidget::windowsHide()
{

}

void MainWidget::on_pushButton_exit_clicked()
{
    if(hardware_thread->isRunning()){
        hardware_thread->stop();
    }
}

void MainWidget::showControlDialog(void)
{
    if(false == control_dialog->isVisible()){
        showMinimized();
        control_dialog->show();
    }else{
        control_dialog->close();
    }
}

void MainWidget::on_hardwareControlFinish()
{
    qDebug() << "The hardware control thread has ended.";
}

void MainWidget::on_pushButton_pulse_clicked()
{
    if(true == hardware_thread->isRunning()){
        QMessageBox::information(NULL,"Error","Please stop the previous task first!");
        return;
    }

    //load params
    ConfigParam config_param;
    config_param.data = &data_matrix[0][0];
    config_param.row = MAX_ROW;
    config_param.col = MAX_COL;

    data_transmission_progress_dialog = new QProgressDialog("data transmission...", "Cancel", 0, 100, this);
    data_transmission_progress_dialog->resize(200,100);
    data_transmission_progress_dialog->show();

    hardware_thread->setConfigParam(config_param);
    hardware_thread->start();
}

void MainWidget::setUTBoardOKState(bool active)
{
    QString style = "border-style: solid;\
            border-width: 3px;\
            border-radius: 46px;\
            border-color: #009844;";
    if(active){
        style += "background-color: #006934;";
    }else{
        style += "background-color: #C1DFB8;";
    }
    this->ui->label_ut_board_ok->setStyleSheet(style);
}

void MainWidget::setAutoCycingState(bool active)
{
    QString style = "border-style: solid;\
            border-width: 3px;\
            border-radius: 46px;\
            border-color: #FAED00;";
    if(active){
        style += "background-color: #FFE300;";
    }else{
        style += "background-color: #FFFCE2;";
    }
    this->ui->label_automatic_cycling->setStyleSheet(style);
}

void MainWidget::updateCycleState(bool enable)
{
    qDebug() << "updateCycleState..." << enable;
    setAutoCycingState(enable);
}

void MainWidget::updatePulseState(bool enable)
{
    qDebug() << "updatePulseState..." << enable;
    setUTBoardOKState(enable);
}

void MainWidget::popupMessage(bool err, QString msg)
{
    qDebug() << msg;
    QMessageBox::information(NULL,true==err?"Error":"Tips",msg);
}

void MainWidget::updateDataPosition(int row,int col)
{
    char stream[200];
    sprintf(stream,"Location (%d,%d)",row+1,col+1);

    this->ui->label_data_position->setText(stream);
}

void MainWidget::on_pushButton_load_data_clicked()
{
    QString filePath = QFileDialog::getOpenFileName(
        this,
        tr("Load Data File"),
        "C:/",
        tr("*.csv")
    );

    if (filePath.isEmpty()) {
        return;
    }

    qDebug() << "select file path:" << filePath;

    LoadData *loader = new LoadData();
    actual_rows = loader->loadCsvData(filePath,&data_matrix[0][0],MAX_ROW,MAX_COL);
    if(0 == actual_rows){
        popupMessage(true,"The input target data dimensions do not match!");
        return;
    }

    if(actual_rows != MAX_ROW){
        popupMessage(true,"The input data file is incomplete!");
        qDebug() << actual_rows;
        return;
    }

    //更新UI控件
    for(int row=0;row<actual_rows;row++)
    {
        for(int col=0;col<MAX_COL;col++)
        {
            channelSelectLayout->setValue(row,col,data_matrix[row][col]);
        }
    }
}

void MainWidget::on_pushButton_open_device_clicked()
{

}

void MainWidget::updateProgressBar(float progress)
{
    if(NULL != data_transmission_progress_dialog){
        data_transmission_progress_dialog->setValue(progress*100);

        if(1 < progress){
            data_transmission_progress_dialog->cancel();
            data_transmission_progress_dialog->close();
            data_transmission_progress_dialog = NULL;
        }
    }
}

 void MainWidget::updateResult(unsigned int off_set,float *result,unsigned int length)
 {
    if(NULL != plot_dialog){
        plot_dialog->refreshData(off_set,result,length);
    }
 }

void MainWidget::createPolylineDialog(void)
{
    if(NULL != plot_dialog){
        //plot_dialog->close();
        plot_dialog = NULL;
    }



    float *data = (float*)malloc(actual_rows*sizeof(float));
    for(int i=0;i<actual_rows;i++){
     data[i] = data_matrix[i][0];
    }

    plot_dialog = new PlotDialog();
    plot_dialog->setData(data,actual_rows);
    plot_dialog->show();

    free(data);
}

