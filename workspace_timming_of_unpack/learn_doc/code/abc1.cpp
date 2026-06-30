#include <stdio.h>

int main(void)
{
  int a[] = {1, 2, 3};
  int count = sizeof(a) / sizeof(a[0]);

  for (int i = 0; i < count; i++) {
    printf("第%d個元素是%d\n", i, a[i]);
  }

  return 0;
}
