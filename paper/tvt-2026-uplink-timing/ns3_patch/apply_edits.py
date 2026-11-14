# One-off generator of the patched sources from orig/ (kept for the record).
h=open('ntn-sib19.h').read()
def rep(s,a,b):
    assert s.count(a)>=1, a[:60]
    return s.replace(a,b)
h=rep(h,"""    /// Part of Te reserved for the UE's own position (GNSS error, motion since the fix).
    void SetUeAllowance(Time a) { m_ueAllowance = a; }
""","""    /// Part of Te reserved for the UE's motion since the fix (the GNSS part is SetGnssBudget).
    void SetUeAllowance(Time a) { m_ueAllowance = a; }
    /// GNSS term of the UE allowance, Proposition 3 (Nov 2026). The UE redraws its fix every
    /// periodS, so over a validity V it uses N = ceil(V / periodS) fixes; the allowance is
    /// 2 z(V) sigma / c with z(V) the two-sided Gaussian quantile of deltaG / N (union bound).
    void SetGnssBudget(double sigmaM, double periodS, double deltaG)
    {
        m_gnssSigma = sigmaM;
        m_gnssPeriod = periodS;
        m_gnssDelta = deltaG;
    }
    /// Norms of the state-vector quantization error the rule charges: half a step per axis of
    /// the PV format by default; the orbital format of EphemerisInfo-r17 leaves far less.
    void SetStateQuantization(double qpNormM, double qvNormMps)
    {
        m_qpNorm = qpNormM;
        m_qvNorm = qvNormMps;
    }
    /// Transparent payload with the reference point at the satellite: the gNB removes the feeder
    /// delay from its own orbit prediction, so one more leg carries the orbit error at age zero.
    void SetFeederLegBudget(bool on) { m_feederLeg = on; }
    /// UE allowance (s) charged for validity v: motion allowance plus the GNSS term.
    double UeAllowanceAt(double v) const;
""")
h=rep(h,"""    Time m_ueAllowance{Seconds(0)};
""","""    Time m_ueAllowance{Seconds(0)};
    double m_gnssSigma{0.0};
    double m_gnssPeriod{1.0};
    double m_gnssDelta{0.0042};
    double m_qpNorm{0.5 * 1.3 * 1.7320508075688772};
    double m_qvNorm{0.5 * 0.06 * 1.7320508075688772};
    bool m_feederLeg{false};
""")
open('ntn-sib19.h','w').write(h)
c=open('ntn-sib19.cc').read()
c=rep(c,'NS_LOG_COMPONENT_DEFINE("NtnSib19");','''// P5_STANDALONE: this copy is compiled into the paper's executable under another class name, so the
// log component and the TypeId must not collide with the ones the library registers.
#ifdef P5_STANDALONE
NS_LOG_COMPONENT_DEFINE("NtnSib19P5");
#else
NS_LOG_COMPONENT_DEFINE("NtnSib19");
#endif''')
c=rep(c,'''        TypeId("ns3::ntnrrc::NtnSib19Broadcaster")
''','''#ifdef P5_STANDALONE
        TypeId("ns3::ntnrrc::NtnSib19BroadcasterP5")
#else
        TypeId("ns3::ntnrrc::NtnSib19Broadcaster")
#endif
''')
c=rep(c,'''            const double ueShare =
                (m_ulRef == UlSyncReference::Satellite ? 1.0 : 1.0 - m_commonShare) *
                    m_te.GetSeconds() -
                m_ueAllowance.GetSeconds();
            for (double v : kEnum)
            {
                const Coefficients c = FitCoefficients(v);
                if (CommonResidual(c, v) <= m_commonShare * m_te.GetSeconds() &&
                    PredictEphemerisError(v) <= ueShare)
''','''            const double ueShare =
                (m_ulRef == UlSyncReference::Satellite ? 1.0 : 1.0 - m_commonShare) *
                m_te.GetSeconds();
            for (double v : kEnum)
            {
                const Coefficients c = FitCoefficients(v);
                if (CommonResidual(c, v) <= m_commonShare * m_te.GetSeconds() &&
                    PredictEphemerisError(v) <= ueShare - UeAllowanceAt(v))
''')
c=rep(c,'''    const double qp = 0.5 * 1.3 * std::sqrt(3.0);
    const double qv = 0.5 * 0.06 * std::sqrt(3.0);
    return (ResidualAcceleration(m_ueModel, r) * v * v +
            2.0 * (qp + qv * v + m_z * (m_odPos + m_odVel * v))) /
           NtnTimingAdvance::kSpeedOfLight;
}
''','''    // A transparent payload with the reference at the satellite adds the feeder leg, predicted at
    // the current time from the determination in force: z sigma' (m_odPos already holds sigma').
    return (ResidualAcceleration(m_ueModel, r) * v * v +
            2.0 * (m_qpNorm + m_qvNorm * v + m_z * (m_odPos + m_odVel * v)) +
            (m_feederLeg ? 2.0 * m_z * m_odPos : 0.0)) /
           NtnTimingAdvance::kSpeedOfLight;
}

double
NtnSib19Broadcaster::UeAllowanceAt(double v) const
{
    double a = m_ueAllowance.GetSeconds();
    if (m_gnssSigma > 0.0)
    {
        // two-sided quantile of deltaG / N by bisection on erfc(z / sqrt 2) = deltaG / N
        const double n = std::max(1.0, std::ceil(v / m_gnssPeriod));
        const double target = m_gnssDelta / n;
        double lo = 0.0;
        double hi = 10.0;
        for (int i = 0; i < 60; ++i)
        {
            const double z = 0.5 * (lo + hi);
            (std::erfc(z / std::sqrt(2.0)) > target ? lo : hi) = z;
        }
        a += 2.0 * 0.5 * (lo + hi) * m_gnssSigma / NtnTimingAdvance::kSpeedOfLight;
    }
    return a;
}
''')
open('ntn-sib19.cc','w').write(c)
e=open('ntn-rrc-ul-timing.cc').read()
e=rep(e,'#include "ns3/ntn-sib19.h"\n','#include "ntn-sib19.h" // patched copy, Nov 2026; must come before the library headers\n')
e=rep(e,'''UeEphemerisModel
ParseModel(const std::string& s)''',open('ex_helpers.inc').read()+'''UeEphemerisModel
ParseModel(const std::string& s)''')
e=rep(e,'''    double closestApproachS = 450.0; // time of the UE's closest approach after start
''','''    double startS = 0.0;             // simulation time (tsince of the TLE) at which the window opens
    double closestApproachS = 450.0; // time of the UE's closest approach after the window opens
''')
e=rep(e,'''    double commonShare = 0.5; // share of Te given to the common-delay polynomial
''','''    double commonShare = 0.5; // share of Te given to the common-delay polynomial
    double implNs = 0.0;      // part of Te kept for downlink timing and RF error at the UE
    double gnssDelta = 0.0042; // delta_g of (C1): GNSS share of the target, union over fixes
    std::string assumedModel = ""; // propagator the network assumes; empty: the UE's own
    std::string ephFormat = "pv";  // EphemerisInfo-r17 choice: pv|orbital
    std::string ueFilter = "none"; // none|ekf: UE smooths decoded state vectors
    double ekfSigmaA = 0.015;      // EKF process noise (m/s^2): the J2 acceleration two-body drops
    std::string gnbFeeder = "ideal"; // satellite reference: ideal|orbit feeder delay removal at gNB
''')
e=rep(e,'''    cmd.AddValue("closestApproachS", "time of closest approach after start (s)", closestApproachS);
''','''    cmd.AddValue("startS", "simulation time at which the window opens (s since TLE epoch)", startS);
    cmd.AddValue("closestApproachS", "time of closest approach after the window opens (s)", closestApproachS);
    cmd.AddValue("implNs", "part of Te kept for downlink timing and RF error (ns)", implNs);
    cmd.AddValue("gnssDelta", "GNSS share of the (C1) target, union over fixes", gnssDelta);
    cmd.AddValue("assumedModel", "propagator the network sizes for (default: the UE's)", assumedModel);
    cmd.AddValue("ephFormat", "EphemerisInfo-r17 choice: pv|orbital", ephFormat);
    cmd.AddValue("ueFilter", "UE smoothing of decoded states: none|ekf", ueFilter);
    cmd.AddValue("ekfSigmaA", "EKF acceleration noise (m/s^2)", ekfSigmaA);
    cmd.AddValue("gnbFeeder", "satellite reference: gNB removes feeder delay ideal|orbit", gnbFeeder);
''')
e=rep(e,'''    sat->GetEcefStateAt(Seconds(closestApproachS), s0, v0);
''','''    sat->GetEcefStateAt(Seconds(startS + closestApproachS), s0, v0);
''')
e=rep(e,'''    Vector gwBase = Rotate(nadir, cross, gwCrossTrackKm * 1e3 / kRe);''','''    const double caLatDeg = std::asin(nadir.z) * 180.0 / M_PI;
    Vector gwBase = Rotate(nadir, cross, gwCrossTrackKm * 1e3 / kRe);''')
e=rep(e,'''            w->AddWaypoint(Waypoint(Seconds(t), Rotate(u0, dir, psi)));
''','''            w->AddWaypoint(Waypoint(Seconds(startS + t), Rotate(u0, dir, psi)));
''')
e=rep(e,'''    sib->SetAssumedUeModel(ParseModel(model));
    sib->SetTimingErrorLimit(NanoSeconds(teNs), commonShare);
''','''    sib->SetAssumedUeModel(ParseModel(assumedModel.empty() ? model : assumedModel));
    const double limitNs = teNs - implNs; // what the extrapolation may use of Te
    sib->SetTimingErrorLimit(NanoSeconds(limitNs), commonShare);
''')
e=rep(e,'''    sib->SetUeAllowance(NanoSeconds(ueAllowanceNs));
''','''    sib->SetUeAllowance(NanoSeconds(ueAllowanceNs));
    sib->SetGnssBudget(gnssSigmaM, gnssPeriodS, gnssDelta);
    const bool orbital = ephFormat == "orbital";
    if (orbital)
    {
        // half a step of each element, as a position and a velocity norm at 7000 km: about 0.2 m
        // and 2e-4 m/s, charged generously
        sib->SetStateQuantization(0.3, 3e-4);
    }
    const bool feederAtGnb = ref != "gateway" && gnbFeeder == "orbit";
    sib->SetFeederLegBudget(feederAtGnb && odBudget);
''')
e=rep(e,'''    uint64_t nViol = 0;

    std::function<void()> sample = [&]() {''','''    uint64_t nViol = 0;
    SatEkf ekf;

    // what the UE decodes: the latest content, with its state vector in the orbital format if
    // asked (from the network's unquantized state at the epoch) and smoothed if the UE filters
    std::pair<Vector, Vector> orbState;
    Callback<void, const Sib19Content&> onBroadcast([&orbState](const Sib19Content&) {
        Vector p;
        Vector v;
        NetworkOrbit(Seconds(0), p, v);
        QuantizeOrbital(p, v);
        orbState = {p, v};
    });
    if (orbital)
    {
        sib->TraceConnectWithoutContext("Broadcast", onBroadcast);
    }

    std::function<void()> sample = [&]() {''')
e=rep(e,'''                pc->ReceiveSib19(sib->GetLatest());
                ++nReads;
''','''                Sib19Content decoded = sib->GetLatest();
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
''')
e=rep(e,'''            const double est = pc->ComputeTa().GetSeconds();
            const double err = (est - truth) * 1e9;
            maxAbs = std::max(maxAbs, std::fabs(err));
            nViol += (std::fabs(err) > teNs) ? 1 : 0;
''','''            double est = pc->ComputeTa().GetSeconds();
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
''')
e=rep(e,'''        const bool visible = eu >= maskDeg && (ref != "gateway" || eg >= maskDeg);''','''        const bool visible = eu >= maskDeg && ((ref != "gateway" && !feederAtGnb) || eg >= maskDeg);''')
e=rep(e,'''        if (Simulator::Now() + MilliSeconds(100) <= Seconds(simTimeS))
''','''        if (Simulator::Now() + MilliSeconds(100) <= Seconds(startS + simTimeS))
''')
e=rep(e,'''    sib->Start();
    Simulator::Schedule(MilliSeconds(50), sample);
    Simulator::Stop(Seconds(simTimeS));
''','''    Simulator::Schedule(Seconds(startS), [&sib]() { sib->Start(); });
    Simulator::Schedule(Seconds(startS) + MilliSeconds(50), sample);
    Simulator::Stop(Seconds(startS + simTimeS));
''')
e=rep(e,'''              << " infeasible=" << sib->GetInfeasibleCount()
''','''              << " infeasible=" << sib->GetInfeasibleCount() << " ca_lat_deg=" << caLatDeg
''')
open('ntn-rrc-ul-timing.cc','w').write(e)
