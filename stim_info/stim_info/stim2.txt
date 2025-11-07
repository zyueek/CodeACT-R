int mysteryOp(int x, int y){
  int result=0;
  while(y>0){
  if(y%2!=0){
  result+=x;}
  x*=2;
  y/=2;
  }
  return result;
}
