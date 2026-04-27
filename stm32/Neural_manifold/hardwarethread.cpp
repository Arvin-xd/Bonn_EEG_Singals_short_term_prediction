#include "event.h"
#include "hardwarethread.h"

#include <QDebug>
#include <QSerialPort>
#include <QSerialPortInfo>

HardwareThread::HardwareThread()
{
    this->run_flag = false;
    serial = NULL;
}

void HardwareThread::setConfigParam(ConfigParam &p)
{
    this->param.row = p.row;
    this->param.col = p.col;
    this->param.data = p.data;

    qDebug() << "row:" << p.row;
    qDebug() << "col:" << p.col;
}

void HardwareThread::stop()
{
    run_flag = false;
}

bool HardwareThread::waitReceipt(unsigned short e)
{
    if(!serial->waitForReadyRead(1000)){
        return false;
    }

    while(this->last_event_type == e){

    };

    return true;
}


void HardwareThread::run()
{
    run_flag = true;

    /* 输出信息 */
    qDebug() << "the sub thread("<< QThread::currentThread() << ") start: ";

    //获取可用串口名到下拉栏
    QList<QSerialPortInfo> list = QSerialPortInfo::availablePorts();
    for(int i=0;i<list.size();i++)
    {
        qDebug() << "portName"<< list.at(i).portName();
        qDebug() << "vender"<< list.at(i).vendorIdentifier();
        qDebug() << "productId"<< list.at(i).productIdentifier();


        //通过设备描述符来检查设备是否存在（自定义描述符，商用需要注册）
        if((53182 == list.at(i).vendorIdentifier())
                && (22336 == list.at(i).productIdentifier())){
          //创建串口对象
          if(NULL != serial){
              serial->close();
              serial = NULL;
          }

          serial = new QSerialPort;
          serial->setPortName(list.at(i).portName());
          serial->setBaudRate(QSerialPort::Baud115200);
          serial->setDataBits(QSerialPort::Data8);
          serial->setParity(QSerialPort::NoParity);
          serial->setStopBits(QSerialPort::OneStop);
          serial->setFlowControl(QSerialPort::NoFlowControl);//无流控制
          if(serial->open(QIODevice::ReadWrite)){
              QObject::connect(serial,&QSerialPort::readyRead,this,&HardwareThread::readStream);

                serial->clear();

                unsigned char *stream;
                unsigned int stream_length;
                Event *event = new Event();
                event->tectonic(EventType::EVENT_CONNECT,NULL,0x00);
                event->toStream((void**)&stream,&stream_length);
                this->last_event_type = event->event_type;

                //https://blog.csdn.net/weixin_44939430/article/details/136328491
                serial->write((char*)stream,stream_length);
                if (!serial->waitForBytesWritten(1000)) {
                    // 处理写入超时或错误
                    emit popupMessage(true,"data send err!");
                    qDebug() << "send err";
                    recovery();
                    return ;
                }

                if(false == waitReceipt(EventType::EVENT_CONNECT)){
                    emit popupMessage(true,"Device No response!");
                    qDebug() << "Device No response!";
                    recovery();
                    return ;
                }

                if(EventType::EVENT_ACK != this->last_event_type){
                    emit popupMessage(true,"Response failed!");
                    qDebug() << "Response failed!" << this->last_event_type;
                    recovery();
                    return ;
                }
            }
        }
    }

    if(NULL == serial){
        emit popupMessage(true,"No device detected!");
        qDebug() << "No device detected!";
        recovery();
        return ;
    }

    emit updateConnectState(true);

    //开始传输数据
    for(int row=0;row<param.row;row++)
    {
        for(int col=0;col<param.col;col+=10)
        {
            unsigned char data[44];
            unsigned short *start_row = (unsigned short*)&data[0];
            unsigned short *start_col = (unsigned short*)&data[2];
            float *value = (float*)&data[4];

            *start_row = row;
            *start_col = col;
            memcpy(value,&param.data[row*param.col+col],10*4);

            unsigned char *stream;
            unsigned int stream_length;
            Event *event = new Event();
            event->tectonic(EventType::EVENT_TRANSFER_DATA,data,sizeof (data));
            event->toStream((void**)&stream,&stream_length);

            for(int i=0;i<10;i++)
            {
                //https://blog.csdn.net/weixin_44939430/article/details/136328491
                this->last_event_type = event->event_type;
                serial->write((char*)stream,stream_length);
                if (!serial->waitForBytesWritten(1000)) {
                    // 处理写入超时或错误
                    emit popupMessage(true,"send error!");
                    qDebug() << "send error!";
                    recovery();
                    return ;
                }

                if(false == waitReceipt(EVENT_TRANSFER_DATA)){
                    qDebug() << "Device No response!";
                }

                if(EventType::EVENT_ACK == this->last_event_type){
                    break;
                }
            }

            //更新进度条
            //qDebug() << "row:" << row << "col:" << col;
            emit updateProgress((1.0*row*param.col+col)/(param.row*param.col));
        }
    }

    emit updateProgress(2);
    emit startExecution();

    qDebug() << "start";

    //开始数据运算
    run_index = false;
    unsigned char *stream;
    unsigned int stream_length;
    Event *event = new Event();
    event->tectonic(EventType::EVENT_START,NULL,0x00);
    event->toStream((void**)&stream,&stream_length);
    this->last_event_type = event->event_type;

    //https://blog.csdn.net/weixin_44939430/article/details/136328491
    serial->write((char*)stream,stream_length);
    if (!serial->waitForBytesWritten(1000)) {
        // 处理写入超时或错误
        emit popupMessage(true,"send error!");
        qDebug() << "send error";
        recovery();
        return ;
    }

    if(false == waitReceipt(EventType::EVENT_START)){
        qDebug() << "Device No response!";
        recovery();
        return ;
    }

    if(EventType::EVENT_ACK != this->last_event_type){
        qDebug() << "Response failed!" << this->last_event_type;
        recovery();
        return ;
    }

    while(run_flag)
    {
        serial->waitForReadyRead(100);
        //QThread::msleep(100);
        //qDebug() << "run";
    }

    recovery();
}

void HardwareThread::recovery()
{
    qDebug() << "thread recovery";

    if(NULL != serial){
        unsigned char *stream;
        unsigned int stream_length;

        Event *event = new Event();
        event->tectonic(EventType::EVENT_STOP,NULL,0x00);
        event->toStream((void**)&stream,&stream_length);

        for(int i=0;i<4;i++)
        {
            serial->write((char*)stream,stream_length);
            if (!serial->waitForBytesWritten(1000)) {
                // 处理写入超时或错误
                emit popupMessage(true,"send error!");
                qDebug() << "send error!";
            }

            if(false == waitReceipt(EventType::EVENT_STOP)){
                qDebug() << "Device No response!";
            }

            if(EventType::EVENT_ACK == this->last_event_type){
                qDebug() << "Response failed!" << this->last_event_type;
                break ;
            }
        }

        QThread::msleep(100);
            serial->close();
            serial = NULL;
    }

    emit updateConnectState(false);
    emit updateRunState(false);
    emit updateProgress(2);
}


void HardwareThread::readStream()
{
//    qDebug() << "read data:";

    unsigned char stream[64];
    unsigned int recv_length =  serial->read((char*)stream,64);

    Event *recv_event = new Event(stream,recv_length);

    if(recv_event->head != EVENT_HEAD){return;}
    this->last_event_type = recv_event->event_type;

//    char msg[32];
//    sprintf(msg,"event:%04x",recv_event->event_type);
//    qDebug()<<msg;
    switch (recv_event->event_type)
    {
        case EventType::EVENT_ACK:{

            break;
        }

        case EventType::EVENT_TRANSFER_RESULT:{
            unsigned char *data = (unsigned char*)recv_event->data;
            unsigned short off_set = *(unsigned short*)data;

            emit updateRunState((run_index++)%2?true:false);
            emit updateResult(off_set,(float*)&data[2],0x02);

            qDebug()<< "offset:" << off_set;
            break;
        }

        case EventType::EVENT_STOP:{
            qDebug()<< "running stop!";
            run_flag = false;
            break;
        }
        default:break;
    }
}

