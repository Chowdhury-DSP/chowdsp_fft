namespace chowdsp::fft::common
{
static int decompose (int n, int* ifac, const int* ntryh)
{
    int nl = n, nf = 0, i, j;
    for (j = 0; ntryh[j]; ++j)
    {
        int ntry = ntryh[j];
        while (nl != 1)
        {
            int nq = nl / ntry;
            int nr = nl - ntry * nq;
            if (nr == 0)
            {
                ifac[2 + nf++] = ntry;
                nl = nq;
                if (ntry == 2 && nf != 1)
                {
                    for (i = 2; i <= nf; ++i)
                    {
                        int ib = nf - i + 2;
                        ifac[ib + 1] = ifac[ib];
                    }
                    ifac[2] = 2;
                }
            }
            else
                break;
        }
    }
    ifac[0] = n;
    ifac[1] = nf;
    return nf;
}

static void rffti1_ps (int n, float* wa, int* ifac)
{
    static constexpr int ntryh[] = { 4, 2, 3, 5, 0 };
    int k1, j, ii;

    int nf = decompose (n, ifac, ntryh);
    float argh = float (2 * M_PI) / (float) n;
    int is = 0;
    int nfm1 = nf - 1;
    int l1 = 1;
    for (k1 = 1; k1 <= nfm1; k1++)
    {
        int ip = ifac[k1 + 1];
        int ld = 0;
        int l2 = l1 * ip;
        int ido = n / l2;
        int ipm = ip - 1;
        for (j = 1; j <= ipm; ++j)
        {
            float argld;
            int i = is, fi = 0;
            ld += l1;
            argld = (float) ld * argh;
            for (ii = 3; ii <= ido; ii += 2)
            {
                i += 2;
                fi += 1;
                wa[i - 2] = std::cos ((float) fi * argld);
                wa[i - 1] = std::sin ((float) fi * argld);
            }
            is += ido;
        }
        l1 = l2;
    }
}

void cffti1_ps (int n, float* wa, int* ifac)
{
    static constexpr int ntryh[] = { 5, 3, 4, 2, 0 };
    int k1, j, ii;

    int nf = decompose (n, ifac, ntryh);
    float argh = float (2 * M_PI) / (float) n;
    int i = 1;
    int l1 = 1;
    for (k1 = 1; k1 <= nf; k1++)
    {
        int ip = ifac[k1 + 1];
        int ld = 0;
        int l2 = l1 * ip;
        int ido = n / l2;
        int idot = ido + ido + 2;
        int ipm = ip - 1;
        for (j = 1; j <= ipm; j++)
        {
            float argld;
            int i1 = i, fi = 0;
            wa[i - 1] = 1;
            wa[i] = 0;
            ld += l1;
            argld = (float) ld * argh;
            for (ii = 4; ii <= idot; ii += 2)
            {
                i += 2;
                fi += 1;
                wa[i - 1] = std::cos ((float) fi * argld);
                wa[i] = std::sin ((float) fi * argld);
            }
            if (ip > 5)
            {
                wa[i1 - 1] = wa[i - 1];
                wa[i1] = wa[i];
            }
        }
        l1 = l2;
    }
}
}
