#include "loaddata.h"
#include <QApplication>
#include <QFile>
#include <QTextStream>
#include <QStringList>
#include <QDebug>
#include <vector>

LoadData::LoadData()
{

}

//// 二维数组类型定义
//typedef double InputData[ROWS][COLS];

/**
 * @brief 加载CSV文件数据到二维数组
 * @param filePath CSV文件路径
 * @param input_data 输出的二维数组
 * @param actualRows 实际读取的行数
 * @return 加载成功返回true，失败返回false
 */
//bool loadCsvData(const QString &filePath, InputData &input_data, int &actualRows)
//{

//}

///**
// * @brief 打印完整的二维数组数据（调试用）
// * @param input_data 二维数组
// * @param rows 实际行数
// * @param cols 列数
// */
//void printFullData(const InputData &input_data, int rows, int cols)
//{
//    qDebug() << "\n=== 完整数据输出 ===";
//    qDebug() << "数据维度:" << rows << "行 x" << cols << "列";

//    for (int i = 0; i < rows; i++) {
//        QString dataLine = QString("行%1: ").arg(i + 1, 3);
//        for (int j = 0; j < cols; j++) {
//            dataLine += QString("%1  ").arg(input_data[i][j], 10, 'f', 6);
//        }
//        qDebug() << dataLine;
//    }
//}

//int main(int argc, char *argv[])
//{
//    QApplication a(argc, argv);

//    // 1. 定义400x20的二维数组
//    InputData input_data = {0};  // 初始化所有元素为0
//    int actualRows = 0;

//    // 2. CSV文件路径（请根据实际情况修改）
//    // 注意：在QT中使用文件路径时，建议使用绝对路径或资源文件
//    QString csvFilePath = "lorenz.csv";  // 当前工作目录下的文件
//    // 或者使用绝对路径，例如：
//    // QString csvFilePath = "C:/data/lorenz.csv";  // Windows
//    // QString csvFilePath = "/home/user/data/lorenz.csv";  // Linux/Mac

//    // 3. 加载CSV数据
//    qDebug() << "开始加载CSV文件:" << csvFilePath;
//    bool loadSuccess = loadCsvData(csvFilePath, input_data, actualRows);

//    if (loadSuccess) {
//        qDebug() << "\n=== 数据加载完成 ===";
//        qDebug() << "数组定义: input_data[" << ROWS << "][" << COLS << "]";
//        qDebug() << "实际数据: " << actualRows << "行 x" << COLS << "列";

//        // 4. 示例：访问数组中的数据
//        qDebug() << "\n=== 数据访问示例 ===";
//        if (actualRows > 0 && COLS > 0) {
//            qDebug() << "input_data[0][0] (第一行第一列):" << input_data[0][0];
//            qDebug() << "input_data[0][19] (第一行第二十列):" << input_data[0][19];

//            if (actualRows > 10) {
//                qDebug() << "input_data[9][9] (第十行第十列):" << input_data[9][9];
//            }

//            qDebug() << "input_data[" << actualRows - 1 << "][" << COLS - 1 << "] (最后一行最后一列):"
//                     << input_data[actualRows - 1][COLS - 1];
//        }

//        // 5. 如果需要打印完整数据（调试时使用）
//        // printFullData(input_data, actualRows, COLS);

//    } else {
//        qCritical() << "数据加载失败，程序退出";
//        return 1;
//    }

//    qDebug() << "\n程序执行完成！";

//    // 如果不需要GUI窗口，可以直接返回0
//    // return a.exec();
//    return 0;
//}

int LoadData::loadCsvData(const QString &filePath, float *input_data,int row, int col)
{
    // 初始化实际行数为0
    int actualRows = 0;

    if(NULL == input_data){return 0;}

    // 打开文件
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qCritical() << "无法打开文件:" << filePath;
        qCritical() << "错误信息:" << file.errorString();
        return 0;
    }

    QTextStream in(&file);
    QString line;
    int rowIndex = 0;

    // 读取文件头（第一行，包含列名）
    if (!in.atEnd()) {
        line = in.readLine();
        qDebug() << "The file header was successfully read, which includes: " << line.split(",").size() << "column";

        // 验证列数是否为指定的
        int real_col = line.split(",").size();
        if (real_col != col) {
            qCritical() << "The number of items does not match! Expectation " << col << "column，reality:" << real_col << "column";
            file.close();
            return false;
        }
    }

    // 读取数据行
    in.seek(0);
    while (!in.atEnd() && (rowIndex < row)) {
        line = in.readLine();
        // 跳过空行
        if (line.trimmed().isEmpty()) {
            continue;
        }

        // 分割CSV数据（使用逗号作为分隔符）
        QStringList dataList = line.split(",");

        // 验证当前行的列数
        if (dataList.size() != col) {
            qWarning() << "The number of data columns in the " << (rowIndex + 1) << "th row does not match. Skip this row.";
            qWarning() << "Current row data:" << line;
            continue;
        }

        // 将字符串数据转换为double并存储到二维数组
        bool rowValid = true;
        for (int colIndex = 0; colIndex < col; colIndex++) {
            bool ok;
            float value = dataList[colIndex].toFloat(&ok);

            if (ok) {
                input_data[rowIndex*col+colIndex] = value;
            } else {
                qWarning() << "Data at row 11" << (rowIndex + 1) << ", column" << (colIndex + 1) << "22 is invalid. Skip this row.";
                qWarning() << "invalid data:" << dataList[colIndex];
                rowValid = false;
                break;
            }
        }

        if (rowValid) {
            rowIndex++;
        }
    }

    // 关闭文件
    file.close();

    // 设置实际读取的行数
    actualRows = rowIndex;

    qDebug() << "Data loading successful. "<<actualRows << " rows and "<< col <<"columns.";

    return actualRows;
    // 输出加载结果
//    if (actualRows > 0) {
//        qDebug() << "数据加载成功！";
//        qDebug() << "实际读取行数:" << actualRows;
//        qDebug() << "读取列数:" << col;
//        qDebug() << "数据数组维度: input_data[" << actualRows << "][" << col << "]";

//        // 显示前5行前5列的数据作为验证
//        qDebug() << "\n前5行前5列数据预览:";
//        int previewRows = qMin(actualRows, 5);
//        int previewCols = qMin(col, 5);

//        for (int i = 0; i < previewRows; i++) {
//            QString previewLine = QString("第%1行: ").arg(i + 1);
//            for (int j = 0; j < previewCols; j++) {
//                previewLine += QString("%1  ").arg(input_data[i*col+j], 10, 'f', 6);
//            }
//            qDebug() << previewLine;
//        }

//        return actualRows;
//    } else {
//        qCritical() << "未读取到任何有效数据！";
//        return 0;
//    }

}

/*
QT项目配置说明：

1. .pro文件配置：
需要在项目的.pro文件中添加以下内容：
QT += core gui widgets
CONFIG += c++11
TARGET = LorenzDataLoader
TEMPLATE = app
SOURCES += main.cpp

2. 文件路径设置：
   - 将lorenz.csv文件放在项目的构建目录下
   - 或者在代码中指定正确的绝对路径
   - 或者将文件添加到QT资源文件(.qrc)中

3. 支持的CSV格式：
   - 逗号分隔符(,)
   - 第一行为列名（会自动跳过）
   - 数据为浮点数格式
   - 支持空行自动跳过
   - 自动验证列数一致性

4. 错误处理：
   - 文件打开失败处理
   - 列数不匹配验证
   - 数据格式错误处理
   - 空数据处理

5. 扩展性：
   - 可通过修改ROWS和COLS常量调整数组大小
   - loadCsvData函数可单独调用
   - 支持数据预览和完整打印
*/
