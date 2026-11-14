// ---- orbital format of EphemerisInfo-r17 (TS 38.331): osculating elements in the inertial frame
// that coincides with ECEF at the epoch, quantized to the field steps and turned back into a state.
constexpr double kMu = 3.986004418e14;
constexpr double kOmegaE = 7.2921150e-5;

void
QuantizeOrbital(Vector& p, Vector& v)
{
    double r[3] = {p.x, p.y, p.z};
    double u[3] = {v.x - kOmegaE * p.y, v.y + kOmegaE * p.x, v.z}; // ECEF velocity -> inertial
    const double rn = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
    const double v2 = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
    const double rv = r[0] * u[0] + r[1] * u[1] + r[2] * u[2];
    double a = 1.0 / (2.0 / rn - v2 / kMu);
    const double hx = r[1] * u[2] - r[2] * u[1];
    const double hy = r[2] * u[0] - r[0] * u[2];
    const double hz = r[0] * u[1] - r[1] * u[0];
    const double hn = std::sqrt(hx * hx + hy * hy + hz * hz);
    double inc = std::acos(hz / hn);
    double raan = std::atan2(hx, -hy);
    const double ex = (v2 / kMu - 1.0 / rn) * r[0] - rv / kMu * u[0];
    const double ey = (v2 / kMu - 1.0 / rn) * r[1] - rv / kMu * u[1];
    const double ez = (v2 / kMu - 1.0 / rn) * r[2] - rv / kMu * u[2];
    double ecc = std::sqrt(ex * ex + ey * ey + ez * ez);
    const double cO = std::cos(raan);
    const double sO = std::sin(raan);
    const double ci = std::cos(inc);
    const double si = std::sin(inc);
    const double nx[3] = {cO, sO, 0.0};
    const double ny[3] = {-sO * ci, cO * ci, si};
    double argp = std::atan2(ex * ny[0] + ey * ny[1] + ez * ny[2], ex * nx[0] + ey * nx[1]);
    const double xp = r[0] * nx[0] + r[1] * nx[1];
    const double yp = r[0] * ny[0] + r[1] * ny[1] + r[2] * ny[2];
    const double nu = std::atan2(yp, xp) - argp;
    double E = 2.0 * std::atan2(std::sqrt(1.0 - ecc) * std::sin(nu / 2.0),
                                std::sqrt(1.0 + ecc) * std::cos(nu / 2.0));
    double M = E - ecc * std::sin(E);
    // field steps of Orbital-r17
    constexpr double kA = 4.249e-3;
    constexpr double kE = 1.431e-8;
    constexpr double kAng = 2.341e-8;
    auto q = [](double x, double s) { return std::round(x / s) * s; };
    a = 6.5e6 + q(a - 6.5e6, kA);
    ecc = q(ecc, kE);
    inc = q(inc, kAng);
    raan = q(raan, kAng);
    argp = q(argp, kAng);
    M = q(M, kAng);
    E = M; // Kepler's equation, then the state
    for (int i = 0; i < 8; ++i)
    {
        E -= (E - ecc * std::sin(E) - M) / (1.0 - ecc * std::cos(E));
    }
    const double b = a * std::sqrt(1.0 - ecc * ecc);
    const double rr = a * (1.0 - ecc * std::cos(E));
    const double n = std::sqrt(kMu / (a * a * a));
    const double xo = a * (std::cos(E) - ecc);
    const double yo = b * std::sin(E);
    const double xd = -a * n * a * std::sin(E) / rr;
    const double yd = b * n * a * std::cos(E) / rr;
    const double cw = std::cos(argp);
    const double sw = std::sin(argp);
    const double cO2 = std::cos(raan);
    const double sO2 = std::sin(raan);
    const double ci2 = std::cos(inc);
    const double si2 = std::sin(inc);
    const double P[3] = {cO2 * cw - sO2 * sw * ci2, sO2 * cw + cO2 * sw * ci2, sw * si2};
    const double Q[3] = {-cO2 * sw - sO2 * cw * ci2, -sO2 * sw + cO2 * cw * ci2, cw * si2};
    p = Vector(xo * P[0] + yo * Q[0], xo * P[1] + yo * Q[1], xo * P[2] + yo * Q[2]);
    const Vector ui(xd * P[0] + yd * Q[0], xd * P[1] + yd * Q[1], xd * P[2] + yd * Q[2]);
    v = Vector(ui.x + kOmegaE * p.y, ui.y - kOmegaE * p.x, ui.z); // inertial -> ECEF velocity
}

// ---- UE-side EKF after Balakrishnan et al. (GLOBECOM 2025): two-body dynamics, first-order state
// transition. Their measurements are range and elevation at 10 ms; a Release 17 UE has only SIB19,
// so each decoded state vector is the measurement here, in an inertial frame fixed at t = 0.
class SatEkf
{
  public:
    void Update(double t, double p[3], double v[3], double sigmaA)
    {
        const double th = kOmegaE * t;
        const double c = std::cos(th);
        const double s = std::sin(th);
        const double vi[3] = {v[0] - kOmegaE * p[1], v[1] + kOmegaE * p[0], v[2]};
        const double z[6] = {c * p[0] - s * p[1], s * p[0] + c * p[1], p[2],
                             c * vi[0] - s * vi[1], s * vi[0] + c * vi[1], vi[2]};
        const double rp = 1.3 * 1.3 / 12.0;
        const double rv = 0.06 * 0.06 / 12.0;
        if (!m_init)
        {
            std::copy(z, z + 6, m_x);
            for (int i = 0; i < 6; ++i)
            {
                std::fill(m_P[i], m_P[i] + 6, 0.0);
                m_P[i][i] = i < 3 ? rp : rv;
            }
            m_init = true;
        }
        else
        {
            Predict(t - m_t, sigmaA);
            double A[6][12]; // Gauss-Jordan on [P + R | P]: right block becomes (P + R)^-1 P = K^T
            for (int i = 0; i < 6; ++i)
            {
                for (int j = 0; j < 6; ++j)
                {
                    A[i][j] = m_P[i][j] + (i == j ? (i < 3 ? rp : rv) : 0.0);
                    A[i][j + 6] = m_P[i][j];
                }
            }
            for (int i = 0; i < 6; ++i)
            {
                int piv = i;
                for (int k = i + 1; k < 6; ++k)
                {
                    if (std::fabs(A[k][i]) > std::fabs(A[piv][i]))
                    {
                        piv = k;
                    }
                }
                std::swap_ranges(A[i], A[i] + 12, A[piv]);
                const double d = A[i][i];
                for (int j = 0; j < 12; ++j)
                {
                    A[i][j] /= d;
                }
                for (int k = 0; k < 6; ++k)
                {
                    if (k != i)
                    {
                        const double f = A[k][i];
                        for (int j = 0; j < 12; ++j)
                        {
                            A[k][j] -= f * A[i][j];
                        }
                    }
                }
            }
            double innov[6];
            for (int i = 0; i < 6; ++i)
            {
                innov[i] = z[i] - m_x[i];
            }
            double nx[6];
            double nP[6][6];
            for (int i = 0; i < 6; ++i)
            {
                nx[i] = m_x[i];
                for (int j = 0; j < 6; ++j)
                {
                    nx[i] += A[j][i + 6] * innov[j];
                    double kp = 0.0;
                    for (int k = 0; k < 6; ++k)
                    {
                        kp += A[k][i + 6] * m_P[k][j];
                    }
                    nP[i][j] = m_P[i][j] - kp;
                }
            }
            std::copy(nx, nx + 6, m_x);
            for (int i = 0; i < 6; ++i)
            {
                for (int j = 0; j < 6; ++j)
                {
                    m_P[i][j] = 0.5 * (nP[i][j] + nP[j][i]);
                }
            }
        }
        m_t = t;
        p[0] = c * m_x[0] + s * m_x[1]; // inertial -> ECEF at t
        p[1] = -s * m_x[0] + c * m_x[1];
        p[2] = m_x[2];
        const double u0 = c * m_x[3] + s * m_x[4];
        const double u1 = -s * m_x[3] + c * m_x[4];
        v[0] = u0 + kOmegaE * p[1];
        v[1] = u1 - kOmegaE * p[0];
        v[2] = m_x[5];
    }

  private:
    void Predict(double dt, double sigmaA)
    {
        const int n = std::max(1, static_cast<int>(std::ceil(dt)));
        const double h = dt / n;
        for (int k = 0; k < n; ++k)
        {
            const double r2 = m_x[0] * m_x[0] + m_x[1] * m_x[1] + m_x[2] * m_x[2];
            const double r = std::sqrt(r2);
            double F[6][6] = {};
            for (int i = 0; i < 6; ++i)
            {
                F[i][i] = 1.0;
            }
            for (int i = 0; i < 3; ++i)
            {
                F[i][i + 3] = h;
                for (int j = 0; j < 3; ++j)
                {
                    F[i + 3][j] =
                        kMu / (r2 * r) * (3.0 * m_x[i] * m_x[j] / r2 - (i == j ? 1.0 : 0.0)) * h;
                }
            }
            double FP[6][6];
            double nP[6][6];
            for (int i = 0; i < 6; ++i)
            {
                for (int j = 0; j < 6; ++j)
                {
                    FP[i][j] = 0.0;
                    for (int l = 0; l < 6; ++l)
                    {
                        FP[i][j] += F[i][l] * m_P[l][j];
                    }
                }
            }
            for (int i = 0; i < 6; ++i)
            {
                for (int j = 0; j < 6; ++j)
                {
                    nP[i][j] = 0.0;
                    for (int l = 0; l < 6; ++l)
                    {
                        nP[i][j] += FP[i][l] * F[j][l];
                    }
                }
            }
            const double q = sigmaA * sigmaA;
            for (int i = 0; i < 3; ++i)
            {
                nP[i][i] += q * h * h * h * h / 4.0;
                nP[i + 3][i + 3] += q * h * h;
                nP[i][i + 3] += q * h * h * h / 2.0;
                nP[i + 3][i] += q * h * h * h / 2.0;
            }
            std::copy(&nP[0][0], &nP[0][0] + 36, &m_P[0][0]);
            auto f = [](const double* y, double* d) {
                const double rr2 = y[0] * y[0] + y[1] * y[1] + y[2] * y[2];
                const double a = -kMu / (rr2 * std::sqrt(rr2));
                d[0] = y[3];
                d[1] = y[4];
                d[2] = y[5];
                d[3] = a * y[0];
                d[4] = a * y[1];
                d[5] = a * y[2];
            };
            double k1[6], k2[6], k3[6], k4[6], t[6];
            f(m_x, k1);
            for (int i = 0; i < 6; ++i)
            {
                t[i] = m_x[i] + 0.5 * h * k1[i];
            }
            f(t, k2);
            for (int i = 0; i < 6; ++i)
            {
                t[i] = m_x[i] + 0.5 * h * k2[i];
            }
            f(t, k3);
            for (int i = 0; i < 6; ++i)
            {
                t[i] = m_x[i] + h * k3[i];
            }
            f(t, k4);
            for (int i = 0; i < 6; ++i)
            {
                m_x[i] += h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
            }
        }
    }

    bool m_init{false};
    double m_t{0.0};
    double m_x[6]{};
    double m_P[6][6]{};
};

