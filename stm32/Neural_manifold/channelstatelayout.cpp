#include "channelstatelayout.h"

QString ChannelStateLayout::getLabelStyle(bool state)
{
    const QString styledEnable = "QLabel {\
            font-family: HarmonyOS Sans SC;\
            font:bold 18px;\
            color: black;\
            background-color:#B6EFAF;\
            border: 1px solid #88A6C9;}";

    const QString styledDisable = "QLabel {\
            font-family: HarmonyOS Sans SC;\
            font:bold 18px;\
            color: black;\
            background-color:#FEFEFE;\
            border: 1px solid #88A6C9;}";

    return state?styledEnable:styledDisable;
}

ChannelStateLayout::ChannelStateLayout()
{
    setContentsMargins(10,10,10,10);
    setVerticalSpacing(10);     // 设置行间距为10像素
    setHorizontalSpacing(10);   // 设置列间距为0像素

    setAlignment(Qt::AlignCenter);
    for(int row=0;row<10;row++)
    {
        for(int column=0;column<8;column++)
        {
            char str[32];
            sprintf(str,"CH%d",row*8+column+1);

            lbs[row][column] = new QLabel(str);
            lbs[row][column]->setStyleSheet(getLabelStyle(false));
            lbs[row][column]->setAlignment(Qt::AlignCenter);
            lbs[row][column]->setFixedSize(75,32);

            addWidget(lbs[row][column],row,column);
            channelStates[row][column] = false;
        }
    }
}

ChannelStateLayout::~ChannelStateLayout()
{

}

void ChannelStateLayout::setChannelState(int channel,bool state)
{
    if(channel >= 80){return;}

    channelStates[channel/8][channel%8] = state;
    lbs[channel/8][channel%8]->setStyleSheet(getLabelStyle(state));
    lbs[channel/8][channel%8]->setAlignment(Qt::AlignCenter);
}

void ChannelStateLayout::clear()
{
    for(int row=0;row<10;row++)
    {
        for(int column=0;column<8;column++)
        {
            channelStates[row][column] = false;
            lbs[row][column]->setStyleSheet(getLabelStyle(false));
            lbs[row][column]->setAlignment(Qt::AlignCenter);
        }
    }
}
