#include "LKH.h"
#include "Segment.h"

GainType Penalty_CTSP_D(void)
{
#ifdef CTSPD_LOW_PRIORITY_FIRST
    Node *N;
    int *Group, *Count, *Seen, n, p, i, g, h;
    GainType BestP = PLUS_INFINITY, P, Removed, Added;

    n = DimensionSaved;
    Group = (int *) malloc(n * sizeof(int));
    Count = (int *) malloc((Groups + 1) * sizeof(int));
    Seen = (int *) malloc((Groups + 1) * sizeof(int));

    for (p = 1; p >= 0; p--) {
        memset(Count, 0, (Groups + 1) * sizeof(int));
        N = Depot;
        for (i = 0; i < n; i++) {
            Group[i] = N->Group;
            Count[Group[i]]++;
            N = p == 1 ? SUCC(N) : PREDD(N);
        }

        memset(Seen, 0, (Groups + 1) * sizeof(int));
        P = 0;
        for (i = 0; i < n; i++) {
            g = Group[i];
            for (h = g + RelaxationLevel + 1; h <= Groups; h++)
                P += Seen[h];
            Seen[g]++;
        }
        if (P < BestP)
            BestP = P;
        if (BestP == 0)
            break;

        for (i = 0; i < n - 1; i++) {
            g = Group[i];
            Removed = Added = 0;
            for (h = 1; h < g - RelaxationLevel; h++)
                Removed += Count[h];
            for (h = g + RelaxationLevel + 1; h <= Groups; h++)
                Added += Count[h];
            P += Added - Removed;
            if (P < BestP) {
                BestP = P;
                if (BestP == 0)
                    break;
            }
            if (BestP == 0)
                break;
        }
        if (BestP == 0)
            break;
    }
    free(Group);
    free(Count);
    free(Seen);
    return BestP;
#else
    Node *N;
    int *Frq, NLoop, p, i;
    GainType P[2] = {0};

    Frq = (int *) malloc((Groups + 1) * sizeof(int));

    for (p = 1; p >= 0; p--) {
        memset(Frq, 0, (Groups + 1) * sizeof(int));
        N = Depot;
        NLoop = 1;
        while (NLoop &&
               (N = p == 1 ? SUCC(N) : PREDD(N)) != Depot) {
            for (i = 1; i < N->Group - RelaxationLevel; i++) {
                P[p] += Frq[i];
                if (P[p] > CurrentPenalty) {
                    NLoop = 0;
                    break;
                }
            }
            Frq[N->Group]++;
        }
    }
    free(Frq);
    return P[0] < P[1] ? P[0] : P[1];
#endif
}
