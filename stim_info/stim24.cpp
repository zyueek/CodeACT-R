long long operate(int n){
  switch (n){
    case 0:
      return 0;
    case 1:
      return 1;
    default:
      long long mystery1=0;
      long long mystery2=1;
      for (int i=2; i<n; ++i){
        long long mystery3=mystery1
        mystery1=mystery2;
        mystery2=mystery3;
      }
      return mystery2;
      }
 }
