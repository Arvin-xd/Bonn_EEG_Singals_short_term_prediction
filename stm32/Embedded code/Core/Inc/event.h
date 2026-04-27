#ifndef _INCLUDE_EVENT_H_
#define _INCLUDE_EVENT_H_

#include <stdbool.h>

#define EVENT_HEAD 0x78A2

/* 事件类型 */
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

/* 事件数据包 */
typedef struct {
	unsigned short head;		//帧头
	unsigned short eventType;	//事件类型
	unsigned char dataLength;	//数据长度
	unsigned char crc;			//校验位
	void* data;					//数据实体指针(内容不定长度)
}Event;

/* 函数接口 */
bool EventTectonic(Event *packet,EventType eventype,
				void *data,unsigned int dataLength);
bool EventTectonicValueOfBytes(Event *packet,
							unsigned char *bytes,unsigned int length);
bool EventProcessing(unsigned char *stream,unsigned int size,bool (*response)(Event*));

#endif /* _INCLUDE_EVENT_H_ */
