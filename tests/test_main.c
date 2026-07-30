#include <stdio.h>

int test_energy_conservation(void);
int test_kepler(void);
int test_angular_momentum(void);
int test_longterm(void);
int test_multibody(void);
int test_equivalence(void);
int test_convergence_all(void);

int main(void){
    int total = 0;

#ifdef NBODY_GPU
    printf("[validation suite - GPU]\n\n");
#else
    printf("[validation suite - CPU]\n\n");
#endif

    total += test_energy_conservation(); printf("\n");
    total += test_kepler(); printf("\n");
    total += test_angular_momentum(); printf("\n");
    total += test_longterm(); printf("\n");
    total += test_multibody(); printf("\n");
    total += test_equivalence(); printf("\n");
    total += test_convergence_all(); printf("\n");

    printf("\n%s (%d failures)\n", total == 0 ? "PASS" : "FAIL", total);
    return total;
}
