#include "main_widget.h"
#include "controldialog.h"
#include "ui_controldialog.h"

#include "QDebug"
ControlDialog::ControlDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::ControlDialog)
{
    ui->setupUi(this);

    setWindowFlag(Qt::WindowStaysOnTopHint,true);
    setWindowFlag(Qt::WindowContextHelpButtonHint,false);
    setWindowFlag(Qt::WindowMaximizeButtonHint,false);
    setFixedSize(width(),height());
    setWindowTitle("Floating Control Dialog");
}

ControlDialog::~ControlDialog()
{
    delete ui;
}

QPushButton *ControlDialog::getPulseButton()
{
    return ui->pushButton_pulse;
}

QPushButton *ControlDialog::getExitButton()
{
    return ui->pushButton_exit;
}

void ControlDialog::setUTBoardOKState(bool active)
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

void ControlDialog::setAutoCycingState(bool active)
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

void ControlDialog::setBoardOKText(QString& text)
{
    ui->label_2->setText(text);
}

void ControlDialog::updateCycleState(bool enable)
{
    setAutoCycingState(enable);
}

void ControlDialog::updatePulseState(bool enable)
{
    setUTBoardOKState(enable);
}
