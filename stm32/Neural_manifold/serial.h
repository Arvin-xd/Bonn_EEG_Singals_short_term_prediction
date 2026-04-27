#ifndef SERIAL_H
#define SERIAL_H

#include <QtSerialPort/QSerialPort>
#include <QtSerialPort/QSerialPortInfo>

#include <QThread>
class Serial : public QThread
{
public:
    Serial();

    void run();
    QSerialPort  *serial;

public slots:
    void readStream();
};

#endif // SERIAL_H
