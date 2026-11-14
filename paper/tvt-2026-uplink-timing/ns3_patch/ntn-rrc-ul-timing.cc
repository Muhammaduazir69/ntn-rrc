/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
// Uplink timing pre-compensation over a real LEO pass (TVT study, Section V-B).
//
// A satellite flies its own TLE through Vallado SGP4 (ntn-constellation). The network side
// (NtnSib19Broadcaster) fills SIB19 every --sibPeriod from that orbit: ECEF state vector and the
// common-delay polynomial for the chosen uplink synchronization reference point, quantized to
// the TS 38.331 field steps. The UE side (NtnUlTimingPrecompensator) decodes SIB19 every
// --readPeriod (or when ul-SyncValidityDuration expires), propagates the satellite with the
// chosen model from its last GNSS fix, and forms its timing advance. Every 100 ms the example
// compares that advance with the true two-way delay over the actual geometry and writes
// time, information age, UE elevation and timing error to --out.
//
//   ./ns3 run "ntn-rrc-ul-timing --tleLine1=... --tleLine2=... --model=kepler-j2 --ref=gateway"

#include "ns3/core-module.h"
#include "ns3/mobility-module.h"
#include "ntn-sib19.h" // patched copy, Nov 2026; must come before the library headers
#include "ns3/ntn-timing-advance.h"
#include "ntn-ul-timing-precompensator.h" // patched copy: propagates before the epoch
#include "ns3/orbital-elements.h"
#include "ns3/sgp4-mobility-model.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>

using namespace ns3;
using namespace ns3::ntnrrc;

namespace
{
constexpr double kC = 299792458.0;
constexpr double kRe = 6378137.0;

Vector
Unit(const Vector& v)
{
    const double n = v.GetLength();
    return Vector(v.x / n, v.y / n, v.z / n);
}

Vector
Cross(const Vector& a, const Vector& b)
{
    return Vector(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

double
Dot(const Vector& a, const Vector& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

/// Point on the sphere of radius kRe at central angle psi from unit vector u, toward unit w.
Vector
Rotate(const Vector& u, const Vector& w, double psi)
{
    return Vector(kRe * (std::cos(psi) * u.x + std::sin(psi) * w.x),
                  kRe * (std::cos(psi) * u.y + std::sin(psi) * w.y),
                  kRe * (std::cos(psi) * u.z + std::sin(psi) * w.z));
}

double
ElevationDeg(const Vector& ground, const Vector& sat)
{
    const Vector up = Unit(ground);
    const Vector los = sat - ground;
    return std::asin(Dot(up, los) / los.GetLength()) * 180.0 / M_PI;
}

// Network orbit-determination error: a radial/along-track/cross-track offset drawn per orbit
// determination (every g_odPeriod seconds) plus a velocity error carried since that determination.
Ptr<ntncon::Sgp4MobilityModel> g_sat;
double g_odPos = 0.0;
double g_odVel = 0.0;
double g_odPeriod = 60.0;
double g_lead = 0.0; // seconds by which content goes on air ahead of its epoch
Ptr<NormalRandomVariable> g_odRng;
std::map<int64_t, std::pair<Vector, Vector>> g_odDraws;

std::pair<Vector, Vector>
OdErrorFor(int64_t k, const Vector& p, const Vector& v)
{
    auto it = g_odDraws.find(k);
    if (it != g_odDraws.end())
    {
        return it->second;
    }
    // unit vectors of the radial, along-track and cross-track frame at the satellite
    const Vector rhat = Unit(p);
    const Vector chat = Unit(Cross(p, v));
    const Vector ahat = Cross(chat, rhat);
    const double er = g_odRng->GetValue(0.0, std::pow(g_odPos / 3.0, 2));
    const double ea = g_odRng->GetValue(0.0, g_odPos * g_odPos);
    const double ec = g_odRng->GetValue(0.0, std::pow(g_odPos / 3.0, 2));
    const double va = g_odRng->GetValue(0.0, g_odVel * g_odVel);
    const Vector dp(er * rhat.x + ea * ahat.x + ec * chat.x, er * rhat.y + ea * ahat.y + ec * chat.y,
                    er * rhat.z + ea * ahat.z + ec * chat.z);
    const Vector dv(va * ahat.x, va * ahat.y, va * ahat.z);
    g_odDraws[k] = {dp, dv};
    return g_odDraws[k];
}

/// The network's prediction of the satellite state at Now() + offset.
void
NetworkOrbit(Time offset, Vector& p, Vector& v)
{
    g_sat->GetEcefStateAt(offset, p, v);
    if (g_odPos <= 0.0 && g_odVel <= 0.0)
    {
        return;
    }
    // the determination in force is the one made at the start of the current period
    const double now = Simulator::Now().GetSeconds();
    const int64_t k = static_cast<int64_t>(std::floor(now / g_odPeriod));
    const auto err = OdErrorFor(k, p, v);
    const double age = now + offset.GetSeconds() - k * g_odPeriod;
    p = Vector(p.x + err.first.x + err.second.x * age, p.y + err.first.y + err.second.y * age,
               p.z + err.first.z + err.second.z * age);
    v = Vector(v.x + err.second.x, v.y + err.second.y, v.z + err.second.z);
}

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

UeEphemerisModel
ParseModel(const std::string& s)
{
    if (s == "linear")
    {
        return UeEphemerisModel::Linear;
    }
    if (s == "gravity-ecef")
    {
        return UeEphemerisModel::GravityEcef;
    }
    if (s == "kepler")
    {
        return UeEphemerisModel::Kepler;
    }
    return UeEphemerisModel::KeplerJ2;
}
} // namespace

int
main(int argc, char* argv[])
{
    std::string tleLine1 =
        "1 44714U 19074B   26274.55916299  .00027027  00000+0  26602-3 0  9994";
    std::string tleLine2 =
        "2 44714  53.1459 272.8136 0002818  85.3455 274.7883 15.65813331380801";
    double startS = 0.0;             // simulation time (tsince of the TLE) at which the window opens
    double closestApproachS = 450.0; // time of the UE's closest approach after the window opens
    double crossTrackKm = 0.0;       // UE offset from the ground track
    double gwCrossTrackKm = 300.0;   // gateway offset from the ground track
    double gwAlongTrackKm = 0.0;     // gateway offset along the track
    double ueSpeedMps = 0.0;         // vehicle speed along the track direction
    double sibPeriodS = 1.0;
    double readPeriodS = 5.0;
    double simTimeS = 900.0;
    double maskDeg = 25.0;
    double gnssSigmaM = 0.0;
    double gnssPeriodS = 1.0;
    std::string model = "kepler-j2";
    std::string ref = "gateway";
    std::string fit = "taylor";
    std::string validity = "budget";
    double fitWindowS = 10.0;
    std::string out = "ul_timing.csv";
    uint32_t seed = 1;
    double odPos = 0.0;       // network orbit error, 1 sigma along track (m)
    double odVel = 0.0;       // network orbit velocity error, 1 sigma (m/s)
    double odPeriod = 60.0;   // orbit determination period (s)
    bool odBudget = true;     // whether the validity rule budgets the orbit error
    bool ueProject = false;   // UE projects its GNSS fix with the measured velocity
    double ueAllowanceNs = 0.0; // part of Te reserved for UE position error and motion
    double teNs = 716.15;     // Te_NTN, FR1-NTN 30/30 kHz (TS 38.133 Table 7.1C.2-1)
    double commonShare = 0.5; // share of Te given to the common-delay polynomial
    double implNs = 0.0;      // part of Te kept for downlink timing and RF error at the UE
    double gnssDelta = 0.0042; // delta_g of (C1): GNSS share of the target, union over fixes
    std::string assumedModel = ""; // propagator the network assumes; empty: the UE's own
    std::string ephFormat = "pv";  // EphemerisInfo-r17 choice: pv|orbital
    std::string ueFilter = "none"; // none|ekf: UE smooths decoded state vectors
    double ekfSigmaA = 0.015;      // EKF process noise (m/s^2): the J2 acceleration two-body drops
    std::string gnbFeeder = "ideal"; // satellite reference: ideal|orbit feeder delay removal at gNB
    bool budgetLegacy = false; // the rule's budget of the previous binary (regression only)
    double gnssVertRatio = 2.0; // vertical over horizontal GNSS error (0.5 before round 3)
    bool ueTxLabel = false;   // UE labels its prediction with its transmit instant, not the frame time
    double leadS = 0.0;       // L: each epoch goes on air L seconds ahead; the UE decodes at a
                              // random point of the last L seconds of its validity and applies it
    bool leadAware = true;    // the fit and the rule's residual cover [-L, V] rather than [0, V]

    CommandLine cmd(__FILE__);
    cmd.AddValue("tleLine1", "TLE line 1 (spaces may be written as _)", tleLine1);
    cmd.AddValue("tleLine2", "TLE line 2 (spaces may be written as _)", tleLine2);
    cmd.AddValue("startS", "simulation time at which the window opens (s since TLE epoch)", startS);
    cmd.AddValue("closestApproachS", "time of closest approach after the window opens (s)", closestApproachS);
    cmd.AddValue("implNs", "part of Te kept for downlink timing and RF error (ns)", implNs);
    cmd.AddValue("gnssDelta", "GNSS share of the (C1) target, union over fixes", gnssDelta);
    cmd.AddValue("assumedModel", "propagator the network sizes for (default: the UE's)", assumedModel);
    cmd.AddValue("ephFormat", "EphemerisInfo-r17 choice: pv|orbital", ephFormat);
    cmd.AddValue("ueFilter", "UE smoothing of decoded states: none|ekf", ueFilter);
    cmd.AddValue("ekfSigmaA", "EKF acceleration noise (m/s^2)", ekfSigmaA);
    cmd.AddValue("gnbFeeder", "satellite reference: gNB removes feeder delay ideal|orbit", gnbFeeder);
    cmd.AddValue("budgetLegacy", "rule budget of the previous binary", budgetLegacy);
    cmd.AddValue("gnssVertRatio", "vertical over horizontal GNSS error", gnssVertRatio);
    cmd.AddValue("ueTxLabel", "UE labels its prediction with its transmit instant", ueTxLabel);
    cmd.AddValue("leadS", "seconds by which each epoch is announced ahead (L)", leadS);
    cmd.AddValue("leadAware", "fit and residual cover [-L, V] instead of [0, V]", leadAware);
    cmd.AddValue("crossTrackKm", "UE cross-track offset (km)", crossTrackKm);
    cmd.AddValue("gwCrossTrackKm", "gateway cross-track offset (km)", gwCrossTrackKm);
    cmd.AddValue("gwAlongTrackKm", "gateway along-track offset (km)", gwAlongTrackKm);
    cmd.AddValue("ueSpeed", "UE ground speed along the track (m/s)", ueSpeedMps);
    cmd.AddValue("sibPeriod", "SIB19 refresh period (s)", sibPeriodS);
    cmd.AddValue("readPeriod", "UE SIB19 re-read period (s)", readPeriodS);
    cmd.AddValue("simTime", "simulated time (s)", simTimeS);
    cmd.AddValue("mask", "elevation mask (deg)", maskDeg);
    cmd.AddValue("gnssSigma", "UE GNSS horizontal 1-sigma error (m)", gnssSigmaM);
    cmd.AddValue("gnssPeriod", "UE GNSS fix period (s)", gnssPeriodS);
    cmd.AddValue("model", "UE propagator: linear|gravity-ecef|kepler|kepler-j2", model);
    cmd.AddValue("ref", "uplink sync reference point: gateway|satellite", ref);
    cmd.AddValue("fit", "SIB19 coefficient fit: taylor|econ|window", fit);
    cmd.AddValue("validity", "ul-SyncValidity rule: slot|budget|fixed (fixed uses --fitWindow)", validity);
    cmd.AddValue("fitWindow", "fit window when validity=slot (s)", fitWindowS);
    cmd.AddValue("out", "output CSV", out);
    cmd.AddValue("seed", "RNG run number", seed);
    cmd.AddValue("teNs", "uplink timing error limit Te_NTN (ns)", teNs);
    cmd.AddValue("odPos", "network orbit error, 1 sigma along track (m)", odPos);
    cmd.AddValue("odVel", "network orbit velocity error, 1 sigma (m/s)", odVel);
    cmd.AddValue("odPeriod", "network orbit determination period (s)", odPeriod);
    cmd.AddValue("odBudget", "validity rule budgets the orbit error", odBudget);
    cmd.AddValue("ueProject", "UE projects its GNSS fix with the measured velocity", ueProject);
    cmd.AddValue("ueAllowanceNs", "part of Te reserved for UE position error and motion (ns)",
                 ueAllowanceNs);
    cmd.AddValue("commonShare", "share of Te for the common-delay polynomial", commonShare);
    cmd.Parse(argc, argv);
    RngSeedManager::SetRun(seed);

    // ---- satellite: Vallado SGP4 from the TLE
    // CommandLine stops a string value at the first space, so lines may be passed with '_'
    std::replace(tleLine1.begin(), tleLine1.end(), '_', ' ');
    std::replace(tleLine2.begin(), tleLine2.end(), '_', ' ');
    ntncon::TleRecord tle;
    tle.line1 = tleLine1;
    tle.line2 = tleLine2;
    Ptr<ntncon::Sgp4MobilityModel> sat = CreateObject<ntncon::Sgp4MobilityModel>();
    NS_ABORT_MSG_UNLESS(sat->SetTle(tle), "TLE did not parse");
    NS_ABORT_MSG_UNLESS(sat->IsUsingSgp4(), "SGP4 backend not active");
    // The position cache defaults to 100 ms. At a LEO range rate of up to 7 km/s that is up to
    // 0.7 km of stale range, several microseconds of delay, so a timing study must not use it.
    sat->SetCacheGranularity(MicroSeconds(1));

    // ---- geometry of the pass: UE near the ground track at closest approach
    Vector s0;
    Vector v0;
    sat->GetEcefStateAt(Seconds(startS + closestApproachS), s0, v0);
    const Vector nadir = Unit(s0);
    const Vector cross = Unit(Cross(nadir, v0));
    const Vector along = Unit(Cross(cross, nadir));
    const Vector uePos0 = Rotate(nadir, cross, crossTrackKm * 1e3 / kRe);
    const double caLatDeg = std::asin(nadir.z) * 180.0 / M_PI;
    Vector gwBase = Rotate(nadir, cross, gwCrossTrackKm * 1e3 / kRe);
    const Vector gwPos = Rotate(Unit(gwBase), along, gwAlongTrackKm * 1e3 / kRe);

    Ptr<MobilityModel> ue;
    if (ueSpeedMps > 0.0)
    {
        // vehicle on a great circle through the UE start point, waypoints every 10 s
        Ptr<WaypointMobilityModel> w = CreateObject<WaypointMobilityModel>();
        const Vector u0 = Unit(uePos0);
        const Vector dir = Unit(along - Vector(u0.x * Dot(along, u0), u0.y * Dot(along, u0),
                                                u0.z * Dot(along, u0)));
        for (double t = 0.0; t <= simTimeS + 10.0; t += 10.0)
        {
            const double psi = ueSpeedMps * (t - closestApproachS) / kRe;
            w->AddWaypoint(Waypoint(Seconds(startS + t), Rotate(u0, dir, psi)));
        }
        ue = w;
    }
    else
    {
        Ptr<ConstantPositionMobilityModel> c = CreateObject<ConstantPositionMobilityModel>();
        c->SetPosition(uePos0);
        ue = c;
    }
    Ptr<ConstantPositionMobilityModel> gw = CreateObject<ConstantPositionMobilityModel>();
    gw->SetPosition(gwPos);

    // ---- network side
    Ptr<NtnTimingAdvance> ta = CreateObject<NtnTimingAdvance>();
    ta->SetUeMobility(ue);
    ta->SetSatelliteMobility(sat);
    ta->SetGatewayMobility(gw);
    ta->SetPayloadMode(ref == "gateway" ? PayloadMode::Transparent : PayloadMode::RegenerativeFull);
    g_sat = sat;
    g_odPos = odPos;
    g_odVel = odVel;
    g_odPeriod = odPeriod;
    g_odRng = CreateObject<NormalRandomVariable>();
    g_odRng->SetStream(1000 + seed);
    ta->SetSatelliteStateProvider(MakeCallback(&NetworkOrbit));

    Ptr<NtnSib19Broadcaster> sib = CreateObject<NtnSib19Broadcaster>();
    sib->SetSatelliteMobility(sat);
    sib->SetTimingAdvance(ta);
    sib->SetPeriod(Seconds(sibPeriodS));
    sib->SetUlSyncReference(ref == "gateway" ? UlSyncReference::Gateway
                                             : UlSyncReference::Satellite);
    sib->SetGatewayPosition(gwPos);
    NS_ABORT_MSG_UNLESS(leadS >= 0.0 && leadS <= startS, "lead outside [0, startS]");
    g_lead = leadS;
    sib->SetLead(leadS, leadAware);
    sib->SetCoefficientFit(fit == "econ"     ? Sib19CoefficientFit::Economized
                           : fit == "window" ? Sib19CoefficientFit::Window
                                             : Sib19CoefficientFit::Taylor);
    sib->SetFitWindow(Seconds(fitWindowS));
    sib->SetValidityRule(validity == "budget"  ? Sib19ValidityRule::ErrorBudget
                         : validity == "fixed" ? Sib19ValidityRule::Fixed
                                               : Sib19ValidityRule::SlotOverDrift);
    sib->SetAssumedUeModel(ParseModel(assumedModel.empty() ? model : assumedModel));
    const double limitNs = teNs - implNs; // what the extrapolation may use of Te
    sib->SetTimingErrorLimit(NanoSeconds(limitNs), commonShare);
    if (odBudget)
    {
        sib->SetOrbitErrorBudget(odPos + odVel * odPeriod, odVel);
    }
    sib->SetUeAllowance(NanoSeconds(ueAllowanceNs));
    sib->SetBudgetForm(budgetLegacy, maskDeg);
    // the line-of-sight deviation of the fix is at most max(sigma_h, sigma_v), Proposition 3
    sib->SetGnssBudget(gnssSigmaM * std::max(1.0, gnssVertRatio), gnssPeriodS, gnssDelta);
    const bool orbital = ephFormat == "orbital";
    if (orbital)
    {
        // half a step of each element, as a position and a velocity norm at 7000 km: about 0.2 m
        // and 2e-4 m/s, charged generously
        sib->SetStateQuantization(0.3, 3e-4);
    }
    const bool feederAtGnb = ref != "gateway" && gnbFeeder == "orbit";
    sib->SetFeederLegBudget(feederAtGnb && odBudget);

    // ---- UE side
    Ptr<NtnUlTimingPrecompensator> pc = CreateObject<NtnUlTimingPrecompensator>();
    pc->SetUeMobility(ue);
    pc->SetEphemerisModel(ParseModel(model));
    pc->SetGnss(Seconds(gnssPeriodS), gnssSigmaM);
    pc->SetGnssVerticalRatio(gnssVertRatio);
    pc->SetTransmitInstantLabel(ueTxLabel);
    pc->SetPositionProjection(ueProject);
    pc->AssignStreams(seed);

    std::ofstream csv(out);
    csv << "t_s,age_s,validity_s,elev_ue_deg,elev_gw_deg,ta_true_us,err_ns,synced\n";
    csv << std::setprecision(10);
    uint64_t nSamples = 0;
    uint64_t nReads = 0;
    double maxAbs = 0.0;
    uint64_t nViol = 0;
    SatEkf ekf;
    // how far before expiry this UE decodes next: anywhere in the lead, so that every tau in
    // [-L, V] gets exercised, and the same SIB19 count as a UE decoding at expiry
    Ptr<UniformRandomVariable> leadRng = CreateObject<UniformRandomVariable>();
    leadRng->SetStream(2000 + seed);
    double earlyS = 0.0;

    // what the UE decodes: the latest content, with its state vector in the orbital format if
    // asked (from the network's unquantized state at the epoch) and smoothed if the UE filters
    std::pair<Vector, Vector> orbState;
    Callback<void, const Sib19Content&> onBroadcast([&orbState](const Sib19Content&) {
        Vector p;
        Vector v;
        NetworkOrbit(Seconds(g_lead), p, v);
        QuantizeOrbital(p, v);
        orbState = {p, v};
    });
    if (orbital)
    {
        sib->TraceConnectWithoutContext("Broadcast", onBroadcast);
    }

    std::function<void()> sample = [&]() {
        const Vector s = sat->GetPosition();
        const Vector p = ue->GetPosition();
        const double eu = ElevationDeg(p, s);
        const double eg = ElevationDeg(gwPos, s);
        const bool visible = eu >= maskDeg && ((ref != "gateway" && !feederAtGnb) || eg >= maskDeg);
        if (visible)
        {
            if (!pc->HasSib19() || !pc->IsUplinkSynchronized() ||
                pc->GetInformationAge().GetSeconds() >= readPeriodS ||
                (leadS > 0.0 &&
                 pc->GetInformationAge().GetSeconds() >= pc->GetHeldValidity().GetSeconds() - earlyS))
            {
                earlyS = leadS > 0.0 ? leadRng->GetValue(0.0, leadS) : 0.0;
                Sib19Content decoded = sib->GetLatest();
                if (orbital)
                {
                    decoded.ephemeris.positionEcefM = orbState.first;
                    decoded.ephemeris.velocityEcefMps = orbState.second;
                }
                if (ueFilter == "ekf")
                {
                    double ps[3] = {decoded.ephemeris.positionEcefM.x,
                                    decoded.ephemeris.positionEcefM.y,
                                    decoded.ephemeris.positionEcefM.z};
                    double vs[3] = {decoded.ephemeris.velocityEcefMps.x,
                                    decoded.ephemeris.velocityEcefMps.y,
                                    decoded.ephemeris.velocityEcefMps.z};
                    ekf.Update(decoded.ephemeris.epoch.GetSeconds(), ps, vs, ekfSigmaA);
                    decoded.ephemeris.positionEcefM = Vector(ps[0], ps[1], ps[2]);
                    decoded.ephemeris.velocityEcefMps = Vector(vs[0], vs[1], vs[2]);
                }
                pc->ReceiveSib19(decoded);
                ++nReads;
            }
            double truth = 2.0 * CalculateDistance(s, p) / kC;
            if (ref == "gateway")
            {
                truth += 2.0 * CalculateDistance(s, gwPos) / kC;
            }
            double est = pc->ComputeTa().GetSeconds();
            if (feederAtGnb)
            {
                // transparent payload, reference at the satellite: the true delay has the feeder
                // leg and the gNB removes it from its own orbit prediction at the current time
                truth += 2.0 * CalculateDistance(s, gwPos) / kC;
                Vector sHat;
                Vector vHat;
                NetworkOrbit(Seconds(0), sHat, vHat);
                est += 2.0 * CalculateDistance(sHat, gwPos) / kC;
            }
            const double err = (est - truth) * 1e9;
            maxAbs = std::max(maxAbs, std::fabs(err));
            nViol += (std::fabs(err) > limitNs) ? 1 : 0;
            csv << Simulator::Now().GetSeconds() << "," << pc->GetInformationAge().GetSeconds()
                << "," << pc->GetHeldValidity().GetSeconds() << "," << eu << "," << eg
                << "," << truth * 1e6 << "," << err << "," << pc->IsUplinkSynchronized() << "\n";
            ++nSamples;
        }
        if (Simulator::Now() + MilliSeconds(100) <= Seconds(startS + simTimeS))
        {
            Simulator::Schedule(MilliSeconds(100), sample);
        }
    };

    // the same epochs at any lead: the first goes on air L before startS
    Simulator::Schedule(Seconds(startS - leadS), [&sib]() { sib->Start(); });
    Simulator::Schedule(Seconds(startS) + MilliSeconds(50), sample);
    Simulator::Stop(Seconds(startS + simTimeS));
    Simulator::Run();
    Simulator::Destroy();

    std::cout << "propagator=" << sat->GetPropagatorName() << " samples=" << nSamples
              << " reads=" << nReads << " max_abs_err_ns=" << maxAbs << " viol=" << nViol
              << " infeasible=" << sib->GetInfeasibleCount() << " ca_lat_deg=" << caLatDeg
              << " out=" << out << "\n";
    return 0;
}
