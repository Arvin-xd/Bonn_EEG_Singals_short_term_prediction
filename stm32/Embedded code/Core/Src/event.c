#include <string.h>

#include "event.h"
#include "neural_manifold.h"

extern bool run_flag;
extern Matrix* input_data;

/**
 *@function 计算校验码
 *@param 	Event *packet	事件数据包
 *@return 	ErrorStatus 执行状态
 */	
static unsigned char calcCheckCode(Event *packet)
{
	if(NULL == packet)return false;
	
	unsigned char sum=0;
	unsigned char *p=(unsigned char*)packet;
	for(int i=0;i<5;i++)sum += p[i];
	
	p = (unsigned char*)packet->data;
	for(int i=0;i<packet->dataLength;i++)sum += p[i];
	
	packet->crc = (~sum)+1;
	
	return packet->crc;
}

unsigned short checkSum(void *stream,unsigned int size)
{
	unsigned short sum=0;
	
	if((NULL==stream) || (0==size)){return sum;}
	
	unsigned char *p = stream;
	for(int i=0;i<size;i++){
		sum += p[i];
	}
	
	return sum;
}

/**
 *@function 事件构造
 *@param 	Event *packet	事件数据包
 *			EventType eventype	事件类型
 *			void *data			数据内容
 *			unsigned int dataLength数据长度
 *@return 	ErrorStatus 执行状态
 */	
bool EventTectonic(Event *packet,EventType eventype,
				void *data,unsigned int dataLength)
{
	//1.非法指针判断
	if(NULL == packet)return false;
	if((0x00 < dataLength) && (NULL == data))return false;
	if(NULL == data)dataLength = 0x00;
	
	//2.构造数据包(注意字节序)
	packet->head = EVENT_HEAD;
	packet->eventType = eventype;
	packet->dataLength = dataLength;
	packet->data = data;
	
	//3.计算校验位
	calcCheckCode(packet);
	
	return true;
}

/**
 *@function 事件构造
 *@param 	Event *packet	事件数据包
 *			unsigned char *bytes字节数组
 *			unsigned int length	字节数组长度
 *@return 	ErrorStatus 执行状态
 */	
bool EventTectonicValueOfBytes(Event *packet,
							unsigned char *bytes,unsigned int length)
{
	//1.指针参数检查
	if((NULL == packet) || (NULL == bytes))return false;
	
	//2.传输校验
	if(length < 0x06)return false;
	
	//3.将字节数组转换为事件数据包
	packet->head = (unsigned short)(bytes[1]<<8|bytes[0]);
    packet->eventType = (unsigned short)(bytes[3]<<8|bytes[2]);
    packet->dataLength = bytes[4];
	if((0x00 != packet->dataLength) && (NULL == packet->data))return false;
	memcpy(packet->data,&bytes[5],packet->dataLength);
	
	//4.校验位验证
	if(bytes[length-1] != calcCheckCode(packet)){return false;}
	if(packet->head != EVENT_HEAD){return false;}	
	
	return true;
}

/**
 *@function bluetooth event processing
 *			Responds to events on the upper computer.
 *@param 	void
 *@return 	void
 */

bool EventProcessing(unsigned char *stream,unsigned int size,bool (*response)(Event*))
{	
	static unsigned char event_data[0x80]={0x00};
	Event packet={.data=event_data};
	
	//将数据包转换为事件帧并判断是否成功
	unsigned char state;
	if(true == EventTectonicValueOfBytes(&packet,stream,size))
	{
		Event return_packet={0x00};												
		
		switch(packet.eventType)
		{		
			case EVENT_CONNECT:{//请求连接
				
				EventTectonic(&return_packet,EVENT_ACK,NULL,0x00);
				break;
			}
				
			case EVENT_START:{//请求开始运算
				run_flag = true;
				EventTectonic(&return_packet,EVENT_ACK,NULL,0x00);
				break;
			}
			
			case EVENT_STOP:{//请求停止运算
				run_flag = false;
				EventTectonic(&return_packet,EVENT_ACK,NULL,0x00);
				break;
			}
			
			case EVENT_ERROR:{//出现错误
				
				EventTectonic(&return_packet,EVENT_ACK,NULL,0x00);
				break;
			}
			
			case EVENT_TRANSFER_DATA:{//传输数据		
				unsigned short *start_row = (void*)((unsigned int)packet.data);
				unsigned short *start_col = (void*)(((unsigned int)packet.data)+2);
				
				memcpy(getMatrixElementPosition(input_data,*start_row,*start_col),
						(void*)(((unsigned int)packet.data)+4),10*4);
				
				EventTectonic(&return_packet,EVENT_ACK,NULL,0x00);
				break;
			}			
		
			default:{
				EventTectonic(&return_packet,EVENT_RESERVE,NULL,0x00);
				break;
			}
		}
		
		if(NULL !=response){/* 回包 */
			response(&return_packet);
		}
	}else{
		return false;
	}
	
	return true;
}
