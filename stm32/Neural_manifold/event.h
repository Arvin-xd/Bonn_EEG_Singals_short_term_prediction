#ifndef EVENT_H
#define EVENT_H

#include <QByteArray>

#define EVENT_HEAD 0x78A2

typedef enum {
    EVENT_CONNECT               = (unsigned short)0x4121,

    EVENT_START                 = (unsigned short)0x5214,

    EVENT_STOP                  = (unsigned short)0x7521,

    EVENT_ERROR                 = (unsigned short)0x2511,

    EVENT_ACK                   = (unsigned short)0x2201,

    EVENT_TRANSFER_DATA         = (unsigned short)0xBF18,

    EVENT_TRANSFER_RESULT       = (unsigned short)0xAF12,

    EVENT_RESERVE = (unsigned short)0xFFFF
}EventType;

typedef enum {
    EXECUTE_OK=0x01,
    EXECUTE_PARAM_ERR=0x03,
    EXECUTE_CMD_RESERVE=0x05,
}EXECUTING_STATE;

class Event
{
public:
    unsigned short head;		//帧头
    unsigned short event_type;	//事件类型
    unsigned char data_length;	//数据长度
    unsigned char crc;			//校验位
    void* data;					//数据实体指针(内容不定长度)
private:
    void* stream;
    unsigned int stream_length;

public:
    Event();
    Event(unsigned char *stream,unsigned int size);
    ~Event(){
        if(NULL != stream){
            free(stream);
        }

        if(NULL != data){
            free(data);
        }
    }

    bool tectonic(EventType even_type,void *data,unsigned int data_length);
    bool toStream(void **stream,unsigned int *length);
};

#endif // EVENT_H
