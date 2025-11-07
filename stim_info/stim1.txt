int mystery (int arr[], int size){
  int expectedSum = arr[0] + (arr[size - 1] * arr[size - 1] + 1) / 2;
  int actualSum = 0;
  for (int i : arr) {
    actualSum += i;
  }
  int result = expectedSum - actualSum;
  return result;
}
