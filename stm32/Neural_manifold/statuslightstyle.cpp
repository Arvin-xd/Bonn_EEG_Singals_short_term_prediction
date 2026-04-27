#include "statuslightstyle.h"

StatusLightStyle::StatusLightStyle()
{

}

QString StatusLightStyle::getChannelActivationStyle()
{
    QString styleDescription ="";
    
    return styleDescription;
}

QString StatusLightStyle::getChannelCloseStyle()
{
    QString styleDescription ="";
    
    return styleDescription;
}

QString StatusLightStyle::getAutoModeActivationStyle()
{
    QString styleDescription ="";
    
    return styleDescription;
}

QString StatusLightStyle::getAutoModeCloseStyle()
{
    QString styleDescription ="";
    
    return styleDescription;
}

QString StatusLightStyle::getPulseOutputStyle()
{
    QString styleDescription = "background-color: #ff0000;\
                                border-style: solid; \        
                                border-width: 3px; \              
                                border-radius: 30px; \            
                                border-color: #009844;";
    return styleDescription;  
}

QString StatusLightStyle::getPulseStopStyle()
{
    QString styleDescription = "background-color: #C1DFB8;\
                                border-style: solid; \        
                                border-width: 3px; \              
                                border-radius: 30px; \            
                                border-color: #009844;";
    return styleDescription;  
}
