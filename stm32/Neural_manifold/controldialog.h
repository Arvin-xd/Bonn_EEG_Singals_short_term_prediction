#ifndef CONTROLDIALOG_H
#define CONTROLDIALOG_H

#include <QDialog>

namespace Ui {
class ControlDialog;
}

class ControlDialog : public QDialog
{
    Q_OBJECT

public:
    explicit ControlDialog(QWidget *parent = nullptr);
    ~ControlDialog();

    QPushButton* getPulseButton();
    QPushButton* getExitButton();

    void setUTBoardOKState(bool active);
    void setAutoCycingState(bool active);
    void setBoardOKText(QString&);

public slots:
    void updateCycleState(bool enable);
    void updatePulseState(bool enable);

private:
    Ui::ControlDialog *ui;
};

#endif // CONTROLDIALOG_H
