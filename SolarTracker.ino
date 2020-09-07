#include <SD.h>
#include <Wire.h>
#include <SPI.h>
#include <Servo.h>
const int chipSelect = 53;
float azimuthArray[]={
42.07209,
61.33088,
64.10745,
71.48811,
77.14761,
84.53214,
97.06117,
111.94798,
125.97188,
129.90099,
132.74594,
149.78885,
150.45993,
167.47157,
171.28288,
181.07640,
191.57906,
194.51567,
202.21297,
211.65334,
218.57405,
220.20354,
227.92377,
230.87239

};
float zenithArray[]={
30.08077,
69.79403,
70.65662,
72.42571,
74.05619,
76.01001,
79.09036,
81.47836,
81.50947,
86.6712,
86.76503,
88.44222,
96.6333,
100.50242,
101.23683,
102.45814,
103.28433,
104.22756,
104.41641,
107.2683,
110.53903,
111.01479,
112.16286,
157.58256


};


int fieldIndex = 0;
Servo zenithServo;
Servo azimuthServo;

void setup(){
    pinMode(53, OUTPUT);
    //Servo 1 and 2
    zenithServo.attach(10);  // attaches the servo on pin 10 to the servo object
    azimuthServo.attach(9);  // attaches the servo on pin 9 to the servo object
    Serial.begin(9600);
  Serial.println();

}
  void loop()
  {
    for (int i=0; i<=23; i++){
        zenithServo.write(zenithArray[i]);
        azimuthServo.write(azimuthArray[i]);
        delay(60UL * 60UL * 1000UL);  //60 minutes each of 60 seconds each of 1000 milliseconds all unsigned longs
//        delay(2000);
    }

  }
