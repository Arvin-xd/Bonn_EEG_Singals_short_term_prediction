#include "configparam.h"
#include "configparammanager.h"

#include <QDebug>

ConfigParamManager::ConfigParamManager()
{

}

QStringList ConfigParamManager::getFrequencyItemList()
{
    static QStringList items;
    if(0x00 == items.size()){
        for(int i=20;i<=1000;i+=10){
            items.append(QString::number(i));
        }
    }
    return items;
}

QStringList ConfigParamManager::getVoltageItemList()
{
    static QStringList items;
    if(0x00 == items.size()){
        for(int i=70;i<=400;i+=10){
            items.append(QString::number(i));
        }
    }
    return items;
}

QStringList ConfigParamManager::getWaveTypeItemList()
{
    static QStringList items;
    if(0x00 == items.size()){
        items.append("spike");
        items.append("square wave");
        items.append("tone burst");
        //items.append("chirp");
    }
    return items;
}

QStringList ConfigParamManager::getLoopIntervalItemList()
{
    static QStringList items;
    if(0x00 == items.size()){        
        items.append("0.1");
        items.append("1");
        items.append("10");
        items.append("100");
        items.append("1000");
        items.append("3600");
    }
    return items;
}

QStringList ConfigParamManager::getChannelLoopIntervalItemList()
{
    static QStringList items;
    if(0x00 == items.size()){
        items.append("0.1");
        items.append("1");
        items.append("10");
        items.append("100");
        items.append("1000");
        items.append("3600");
    }
    return items;
}

QStringList ConfigParamManager::getBurstCountItemList()
{
    static QStringList items;
    if(0x00 == items.size()){
        for(int i=0;i<16;i++)
        {
            items.append(QString::number(i+1));
        }
    }
    return items;
}

float ConfigParamManager::toFrequencyParam(int item_index)
{
    return (item_index*10+20)/1000.0;
}

float ConfigParamManager::toVoltageParam(int item_index)
{
    return item_index*10+70;
}


unsigned short ConfigParamManager::toBurstCountParam(int item_index)
{
    return (unsigned short)(item_index+1);
}
