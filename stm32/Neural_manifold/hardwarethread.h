#ifndef HARDWARETHREAD_H
#define HARDWARETHREAD_H

#include "configparam.h"

#include <QThread>
#include <QList>
#include <QSerialPort>

class HardwareThread : public QThread
{
    Q_OBJECT

public:
    HardwareThread();

    void setConfigParam(ConfigParam &);
    void stop();



protected:
    void run();

private:
    void recovery(void);
    bool waitReceipt(unsigned short e);
private:
    QList<int> channels;
    ConfigParam param;
    QSerialPort  *serial;
    volatile bool run_flag;
    volatile unsigned short last_event_type;

    int run_index;
    int mux_type;
signals:
    void updateProgress(float progress);
    void updateConnectState(bool state);
    void updateRunState(bool state);
    void updateResult(unsigned int off_set,float *result,unsigned int length);
    void popupMessage(bool err,QString msg);
    void startExecution();

public slots:
    void readStream();
};

#endif // HARDWARETHREAD_H
