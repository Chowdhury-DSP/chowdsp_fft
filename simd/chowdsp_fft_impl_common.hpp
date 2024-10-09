/**
Copyright (c) 2024 Jatin Chowdhury ( jatin@chowdsp.com )
Copyright (c) 2013  Julien Pommier ( pommier@modartt.com )

Based on original fortran 77 code from FFTPACKv4 from NETLIB,
authored by Dr Paul Swarztrauber of NCAR, in 1985.

As confirmed by the NCAR fftpack software curators, the following
FFTPACKv5 license applies to FFTPACKv4 sources. My changes are
released under the same terms.

FFTPACK license:
http://www.cisl.ucar.edu/css/software/fftpack5/ftpk.html

Copyright (c) 2004 the University Corporation for Atmospheric
Research ("UCAR"). All rights reserved. Developed by NCAR's
Computational and Information Systems Laboratory, UCAR,
www.cisl.ucar.edu.

Redistribution and use of the Software in source and binary forms,
with or without modification, is permitted provided that the
following conditions are met:

- Neither the names of NCAR's Computational and Information Systems
  Laboratory, the University Corporation for Atmospheric Research,
  nor the names of its sponsors or contributors may be used to
  endorse or promote products derived from this Software without
  specific prior written permission.

- Redistributions of source code must retain the above copyright
  notices, this list of conditions, and the disclaimer below.

- Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions, and the disclaimer below in the
  documentation and/or other materials provided with the
  distribution.

THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING, BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES OR OTHER LIABILITY, WHETHER IN AN
ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH THE
SOFTWARE.
*/

namespace chowdsp::fft::common
{
static inline int decompose (int n, int* ifac, const int* ntryh)
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

static inline void rffti1_ps (int n, float* wa, int* ifac)
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

void inline cffti1_ps (int n, float* wa, int* ifac)
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
