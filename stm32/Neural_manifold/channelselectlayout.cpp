#include "buttonstyle.h"
#include "channelselectlayout.h"

#include <QDebug>
#include <QSignalMapper>

ChannelSelectLayout::ChannelSelectLayout()
{
    setContentsMargins(10,0,10,0);
    setVerticalSpacing(10); // 设置行间距为10像素
    setHorizontalSpacing(0); // 设置列间距为0像素

    channelSelectList = new QList<int>();
    signelMapper = new QSignalMapper(this);

    for(int row=0;row<400;row++)
    {
        for(int column=0;column<20;column++)
        {            
            btns[row][column] = new QPushButton("-");
            btns[row][column]->setStyleSheet(ButtonStyle::btChannelSelectInactiveStyle());
            btns[row][column]->setFixedSize(70,64);
            btns[row][column]->setFocusPolicy(Qt::ClickFocus);

            signelMapper->setMapping(btns[row][column],row*20+column);
            connect(btns[row][column],SIGNAL(clicked()),signelMapper,SLOT(map()));

            addWidget(btns[row][column],row,column);
        }
    }

    last_position = 0;
    connect(signelMapper,SIGNAL(mapped(int)),this,SLOT(onChannalSelectButtonClick(int)));
}

ChannelSelectLayout::~ChannelSelectLayout()
{

}

void ChannelSelectLayout::onChannalSelectButtonClick(int channel)
{
    if(channel > 400*20){return;}
    int row = channel/20;
    int column = channel%20;

   btns[last_position/20][last_position%20]->setStyleSheet(ButtonStyle::btChannelSelectInactiveStyle());
   last_position = channel;

    btns[row][column]->setStyleSheet(ButtonStyle::btChannelSelectActivetaStyle());

    emit updateDataPosition(row,column);
}

void ChannelSelectLayout::setValue(int row,int col,float value)
{
    if((row < 400) || (col<20)){
        btns[row][col]->setText(QString::number(value));
    }
}
