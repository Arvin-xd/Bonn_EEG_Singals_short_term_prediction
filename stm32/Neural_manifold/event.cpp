#include "event.h"
#include "QDebug"
Event::Event()
{
    this->stream = NULL;
    this->stream_length = 0x00;
    this->head = 0x00;
    this->event_type = EVENT_RESERVE;
    this->data_length = 0x00;
    this->crc = 0x00;
    this->data = NULL;
}

Event::Event(unsigned char *stream,unsigned int size)
{
    this->stream = NULL;
    this->stream_length = 0x00;
    this->head = 0x00;
    this->event_type = EVENT_RESERVE;
    this->data_length = 0x00;
    this->crc = 0x00;
    this->data = malloc(64);

    //1.指针参数检查
    if(NULL == stream) return;

    //2.传输校验
    if(size < 0x06)return;

    //3.将字节数组转换为事件数据包
    head = (unsigned short)(stream[1]<<8|stream[0]);
    event_type = (unsigned short)(stream[3]<<8|stream[2]);
    data_length = stream[4];
    if((0x00 != data_length) && (NULL == data))return;
    memcpy(data,&stream[5],data_length);

    //4.校验位验证
   // if(stream[size-1] != calcCheckCode(this)){return;}
    if(head != EVENT_HEAD){return;}
}

/**
 *@function 计算校验码
 *@param 	Event *packet	事件数据包
 *@return 	ErrorStatus 执行状态
 */
static unsigned char calcCheckCode(Event *packet)
{
    if(NULL == packet)return 0x00;

    unsigned char sum=0;
    unsigned char *p=(unsigned char*)packet;
    for(int i=0;i<5;i++)sum += p[i];

    p = (unsigned char*)packet->data;
    for(int i=0;i<packet->data_length;i++)sum += p[i];

    packet->crc = (~sum)+1;

    return packet->crc;
}

unsigned short checkSum(void *stream,unsigned int size)
{
    unsigned short sum=0;

    if((NULL==stream) || (0==size)){return sum;}

    unsigned char *p = (unsigned char*)stream;
    for(unsigned int i=0;i<size;i++){
        sum += p[i];
    }

    return sum;
}

bool Event::tectonic(EventType even_type,void *data,unsigned int data_length)
{
    //1.非法指针判断
    if((0x00 < data_length) && (NULL == data))return false;
    if(NULL == data)data_length = 0x00;

    //2.构造数据包(注意字节序)
    this->head = EVENT_HEAD;
    this->event_type = even_type;
    this->data_length = data_length;
    this->data = data;

    //3.计算校验位
    calcCheckCode(this);

    return true;
}

bool Event::toStream(void **s,unsigned int *l)
{
    unsigned int packet_length = data_length+5+1;

    if(NULL == this->stream)
    {
        this->stream = malloc(packet_length);
        this->stream_length = packet_length;
    }else if(packet_length != this->stream_length){
        this->stream = realloc(stream,packet_length);
        this->stream_length = packet_length;
    }

    unsigned char *p = (unsigned char*)stream;
    p[1] = (head>>8)&0xff;
    p[0] = head&0xff;

    p[3] = (event_type>>8)&0xff;
    p[2] = event_type&0xff;

    p[4] = data_length;

    memcpy(&p[5],data,data_length);

    p[data_length+5] = crc;

    *s = stream;
    *l = stream_length;

    return true;
}

