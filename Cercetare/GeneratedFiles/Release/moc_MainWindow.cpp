/****************************************************************************
** Meta object code from reading C++ file 'MainWindow.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.8.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../MainWindow.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'MainWindow.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.8.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_MainWindow_t {
    QByteArrayData data[20];
    char stringdata0[337];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_MainWindow_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_MainWindow_t qt_meta_stringdata_MainWindow = {
    {
QT_MOC_LITERAL(0, 0, 10), // "MainWindow"
QT_MOC_LITERAL(1, 11, 19), // "OnSwitchViewClicked"
QT_MOC_LITERAL(2, 31, 0), // ""
QT_MOC_LITERAL(3, 32, 15), // "OnZoomInClicked"
QT_MOC_LITERAL(4, 48, 16), // "OnZoomOutClicked"
QT_MOC_LITERAL(5, 65, 18), // "OnZoomResetClicked"
QT_MOC_LITERAL(6, 84, 18), // "OnToggleLogClicked"
QT_MOC_LITERAL(7, 103, 18), // "OnLoadImageClicked"
QT_MOC_LITERAL(8, 122, 18), // "OnSaveImageClicked"
QT_MOC_LITERAL(9, 141, 14), // "OnSobelClicked"
QT_MOC_LITERAL(10, 156, 21), // "OnGaussianBlurClicked"
QT_MOC_LITERAL(11, 178, 19), // "OnSharpeningClicked"
QT_MOC_LITERAL(12, 198, 23), // "OnColorSmoothingClicked"
QT_MOC_LITERAL(13, 222, 15), // "OnKMeansClicked"
QT_MOC_LITERAL(14, 238, 18), // "OnGrayscaleClicked"
QT_MOC_LITERAL(15, 257, 15), // "OnResizeClicked"
QT_MOC_LITERAL(16, 273, 18), // "OnThresholdClicked"
QT_MOC_LITERAL(17, 292, 12), // "OnSOMClicked"
QT_MOC_LITERAL(18, 305, 12), // "OnSIFTClcked"
QT_MOC_LITERAL(19, 318, 18) // "OnBenchmarkClicked"

    },
    "MainWindow\0OnSwitchViewClicked\0\0"
    "OnZoomInClicked\0OnZoomOutClicked\0"
    "OnZoomResetClicked\0OnToggleLogClicked\0"
    "OnLoadImageClicked\0OnSaveImageClicked\0"
    "OnSobelClicked\0OnGaussianBlurClicked\0"
    "OnSharpeningClicked\0OnColorSmoothingClicked\0"
    "OnKMeansClicked\0OnGrayscaleClicked\0"
    "OnResizeClicked\0OnThresholdClicked\0"
    "OnSOMClicked\0OnSIFTClcked\0OnBenchmarkClicked"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_MainWindow[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
      18,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    0,  104,    2, 0x08 /* Private */,
       3,    0,  105,    2, 0x08 /* Private */,
       4,    0,  106,    2, 0x08 /* Private */,
       5,    0,  107,    2, 0x08 /* Private */,
       6,    0,  108,    2, 0x08 /* Private */,
       7,    0,  109,    2, 0x08 /* Private */,
       8,    0,  110,    2, 0x08 /* Private */,
       9,    0,  111,    2, 0x08 /* Private */,
      10,    0,  112,    2, 0x08 /* Private */,
      11,    0,  113,    2, 0x08 /* Private */,
      12,    0,  114,    2, 0x08 /* Private */,
      13,    0,  115,    2, 0x08 /* Private */,
      14,    0,  116,    2, 0x08 /* Private */,
      15,    0,  117,    2, 0x08 /* Private */,
      16,    0,  118,    2, 0x08 /* Private */,
      17,    0,  119,    2, 0x08 /* Private */,
      18,    0,  120,    2, 0x08 /* Private */,
      19,    0,  121,    2, 0x08 /* Private */,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,

       0        // eod
};

void MainWindow::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        MainWindow *_t = static_cast<MainWindow *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->OnSwitchViewClicked(); break;
        case 1: _t->OnZoomInClicked(); break;
        case 2: _t->OnZoomOutClicked(); break;
        case 3: _t->OnZoomResetClicked(); break;
        case 4: _t->OnToggleLogClicked(); break;
        case 5: _t->OnLoadImageClicked(); break;
        case 6: _t->OnSaveImageClicked(); break;
        case 7: _t->OnSobelClicked(); break;
        case 8: _t->OnGaussianBlurClicked(); break;
        case 9: _t->OnSharpeningClicked(); break;
        case 10: _t->OnColorSmoothingClicked(); break;
        case 11: _t->OnKMeansClicked(); break;
        case 12: _t->OnGrayscaleClicked(); break;
        case 13: _t->OnResizeClicked(); break;
        case 14: _t->OnThresholdClicked(); break;
        case 15: _t->OnSOMClicked(); break;
        case 16: _t->OnSIFTClcked(); break;
        case 17: _t->OnBenchmarkClicked(); break;
        default: ;
        }
    }
    Q_UNUSED(_a);
}

const QMetaObject MainWindow::staticMetaObject = {
    { &QMainWindow::staticMetaObject, qt_meta_stringdata_MainWindow.data,
      qt_meta_data_MainWindow,  qt_static_metacall, Q_NULLPTR, Q_NULLPTR}
};


const QMetaObject *MainWindow::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *MainWindow::qt_metacast(const char *_clname)
{
    if (!_clname) return Q_NULLPTR;
    if (!strcmp(_clname, qt_meta_stringdata_MainWindow.stringdata0))
        return static_cast<void*>(const_cast< MainWindow*>(this));
    return QMainWindow::qt_metacast(_clname);
}

int MainWindow::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 18)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 18;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 18)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 18;
    }
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
