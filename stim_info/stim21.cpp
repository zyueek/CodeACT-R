#include<iostream>
int afunc(int num1, int num2){
    if (num1==0){
        return num2+num1;
    }else if(num2==0){
        return num1*2
    }else{
        return afunc(num1-1, num2-1);
        }
    }
