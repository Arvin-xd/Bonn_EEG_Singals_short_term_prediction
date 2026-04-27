#ifndef CONFIGPARAMMANAGER_H
#define CONFIGPARAMMANAGER_H

#include <QStringList>



class ConfigParamManager
{
    public:
        ConfigParamManager();

        static QStringList getFrequencyItemList(void);
        static QStringList getVoltageItemList(void);
        static QStringList getWaveTypeItemList(void);
        static QStringList getBurstCountItemList(void);
        static QStringList getLoopIntervalItemList(void);
        static QStringList getChannelLoopIntervalItemList(void);

        static float toFrequencyParam(int item_index);
        static float toVoltageParam(int item_index);
        static unsigned short toWaveTypeParam(int item_index);
        static unsigned short toBurstCountParam(int item_index);
};

#endif // CONFIGPARAMMANAGER_H
