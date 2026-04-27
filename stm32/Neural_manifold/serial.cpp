#include "serial.h"
#include <QDebug>

#include "event.h"

Serial::Serial()
{

}

void Serial::run()
{
    //CH343 测试
    //Physical Acoustics
    //Multiplexer

    //6790
    //21970

    serial = new QSerialPort;

    //获取可用串口名到下拉栏
    QList<QSerialPortInfo> list = QSerialPortInfo::availablePorts();


  //  qDebug() << CH343PT_GetVersion();

    //创建串口对象
    serial->setPortName("COM6");
    serial->setBaudRate(QSerialPort::Baud115200);
    serial->setDataBits(QSerialPort::Data8);
    serial->setParity(QSerialPort::NoParity);
    serial->setStopBits(QSerialPort::OneStop);
    serial->setFlowControl(QSerialPort::NoFlowControl);//无流控制
    bool info = serial->open(QIODevice::ReadWrite);

    unsigned char data[10]={1,0,0,0,0,0,0,0,0,0};
    unsigned int index=1;
    if(info == true)
    {
        qDebug() << "open serial" << endl;

        QObject::connect(serial,&QSerialPort::readyRead,this,&Serial::readStream);
        msleep(2000);
        while (1) {

            msleep(300);
            memset (data,0x00,sizeof (data));
            data[9-(index-1)/8] = 1<<(index-1)%8;
            index++;
            unsigned char *stream;
            unsigned int stream_length;
            Event *event = new Event();
            event->tectonic(EventType::EVENT_SET_RELAY_STATE,data,sizeof (data));

            event->toStream((void**)&stream,&stream_length);
            for(int i=0;i<stream_length;i++)
            {
                qDebug() << QString::number(stream[i],16);
            }

            //https://blog.csdn.net/weixin_44939430/article/details/136328491
            serial->write((char*)stream,stream_length);
            if (!serial->waitForBytesWritten(1000)) {
                // 处理写入超时或错误
                qDebug() << "send err";
            }

          //  qDebug() << "send data";
           // return;

           // data[0] <<= 1;


        }

    }

//    if(CH343PT_HandleIsCH34x(serial))
//    {
//        qDebug() << "open serial ok";
//    }else{
//        qDebug() << "open serial false";
//    }

}


void Serial::readStream()
{
    qDebug() << "read";
    qDebug() << serial->readAll();
}
