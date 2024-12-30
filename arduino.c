#include <Arduino.h>
#include <U8g2lib.h>
#include "HX711.h" //HX711로드셀 엠프 관련함수 호출
#define calibration_factor -7050.0 // 로드셀 스케일 값 선언
#define DOUT1  3 //엠프 데이터 아웃 핀 넘버 선언
#define CLK1  2  //엠프 클락 핀 넘버 
#define DOUT2  5 //엠프 데이터 아웃 핀 넘버 선언
#define CLK2  4  //엠프 클락 핀 넘버   
#define DISTANCE  20.0  //무게센서 간격(cm)   
#define OVERSPEED  8.0
#ifdef U8X8_HAVE_HW_SPI
#include <SPI.h>
#endif

U8G2_ST7920_128X64_1_SW_SPI u8g2(U8G2_R0, /* clock=*/ 13, /* data=*/ 11, /* CS=*/ 10, /* reset=*/ 8);
/* VCC -> 빨간색 GND, 갈색 VCC
 * RS/E -> 검정색 RS(CS) , 흰색 R/W(DATA), 회색 E(clock)
 * RST -> 초록색 PSB , 갈색 RST(reset)
 * BLK -> 주황색 BLA , 노란색 BLK
 */

HX711 scale1(DOUT1, CLK1); //엠프 핀 선언 
HX711 scale2(DOUT2, CLK2); //엠프 핀 선언 
const int rainSensor = A4;

void setup(void) {
  u8g2.begin();
  u8g2.enableUTF8Print();    // enable UTF8 support for the Arduino print() function
  Serial.begin(9600);
  scale1.set_scale(calibration_factor);  //스케일 지정 
  scale1.tare();  //스케일 설정
  scale2.set_scale(calibration_factor);  //스케일 지정 
  scale2.tare();  //스케일 설정
  pinMode(rainSensor,INPUT);      // rain sensor
}

void sendSpeed(double speed) {
  Serial.println(speed, 6);
}

void draw1(double speed)
{
  u8g2.setFont(u8g2_font_unifont_t_korean1);
  u8g2.setFontDirection(0);
  u8g2.setCursor(37, 15);
  u8g2.print("Caution");
  u8g2.setCursor(23, 35);
  u8g2.print("Front Bump");
  u8g2.setCursor(0, 55);
  u8g2.print("Speed : ");
  u8g2.setCursor(60, 55);
  u8g2.print(speed, 1);
  u8g2.setCursor(95, 55);
  u8g2.print("cm/s");
}

void draw2(double speed)
{
  u8g2.setFont(u8g2_font_unifont_t_korean1);
  u8g2.setFontDirection(0);
  u8g2.setCursor(37, 15);
  u8g2.print("Caution");
  u8g2.setCursor(34, 35);
  u8g2.print("Wet Road");
  u8g2.setCursor(0, 55);
  u8g2.print("Speed : ");
  u8g2.setCursor(60, 55);
  u8g2.print(speed, 1);
  u8g2.setCursor(95, 55);
  u8g2.print("cm/s");
}

double senseWeight() {
  unsigned long start_time, end_time;
  double duration = 0;
  int weightValue2 = 0;
  int weightValue1 = scale1.get_units();
  if (weightValue1 !=0){
    start_time = millis();

    scale2.tare();  //스케일 설정
    while(weightValue2==0){
      weightValue2 = scale2.get_units();
      delay(100);
    }
    end_time = millis();
    duration = (end_time-start_time);
    duration = duration/1000.0;
    double speed = DISTANCE / duration;
    sendSpeed(speed);
    scale1.tare();  //스케일 설정
    return speed;
    delay(1000);
  }
  delay(100);
}

void loop() {
  // 비 감지 센서 값 읽기
  int rainValue = analogRead(rainSensor);
  u8g2.firstPage();

  double speed = senseWeight();
  if (speed >0){
  // 비가 내리면 (센서 값이 700 미만일 때)
  if (rainValue < 700) {
    u8g2.firstPage();        // 새로운 페이지 시작
    do {
      draw2(speed);               // 비오기전 안내 메시지 출력
    } while (u8g2.nextPage()); // 화면 업데이트
  } else {
    u8g2.firstPage();        // 새로운 페이지 시작
    do {
      draw1(speed);               // 비올때 메시지 출력
    } while (u8g2.nextPage()); // 화면 업데이트
  }
   delay(100);  
  }
}