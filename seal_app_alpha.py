import streamlit as st
import math
import matplotlib.pyplot as plt

def calculate_tmin(T_jaw_C, T0_C, torque, r_lever, A_cm2, d_micro, n_plies,
                   rho, cp, k, Lf, dT_seal, h0, alpha_p,
                   h_air, eps, delta_t, Rq, h_int,
                   include_jaw_mass, m_jaw, cp_jaw):

    sigma = 5.67e-8  # Stefan-Boltzmann constant

    # Unit conversions
    A_m2 = A_cm2 / 1e4
    d = d_micro * 1e-6
    d_total = d * n_plies
    T_jaw = T_jaw_C + 273.15
    T0 = T0_C + 273.15
    F = torque / r_lever  # Force [N]
    p = F / A_m2          # Pressure [Pa]

    # Energy required per unit area
    Q_req = rho * d * (cp * dT_seal + Lf)

    # Contact conductance
    h_contact = h0 + alpha_p * p

    # Heat flux in
    q_in = h_contact * (T_jaw - T0)

    # Heat losses
    q_loss_conv = h_air * (T_jaw - T0)
    q_loss_rad = eps * sigma * (T_jaw**4 - T0**4)
    q_loss_total = q_loss_conv + q_loss_rad

    # Net heat flux
    q_net = q_in - q_loss_total

    # Energy-limited time
    t_energy = (Q_req) / q_net if q_net > 0 else float("inf")

    # Conduction correction
    alpha = k / (rho * cp)
    t_cond = d_total**2 / alpha

    # Interface correction
    t_interface = (1 / h_int) * (d / k)

    # Cooling correction
    t_cool = d_total**2 / (math.pi**2 * alpha)

    # Jaw finite heat capacity correction
    if include_jaw_mass:
        deltaT_jaw = (Q_req * A_m2) / (m_jaw * cp_jaw)
        T_eff = T_jaw - deltaT_jaw
        q_in_eff = h_contact * (T_eff - T0)
        q_net_eff = q_in_eff - q_loss_total
        t_energy = (Q_req) / q_net_eff if q_net_eff > 0 else float("inf")

    # Corrections factors
    f_thickness = 1 + delta_t
    f_surface = 1 / (1 + Rq)

    # Final time
    t_min = f_thickness * f_surface * (t_energy + t_cond + t_interface + t_cool)

    return t_min


def main():
    st.title("📦 Interactive Minimum Seal Time Calculator for LLDPE")

    # ---------------- Jaw & Mechanical Setup ----------------
    st.header("Jaw & Mechanical Setup")
    T_jaw_C = st.number_input("Jaw temperature [°C]", value=240.0)
    T0_C = st.number_input("Ambient temperature [°C]", value=25.0)
    torque = st.slider("Applied torque [N·m]", 20, 2000, 100)
    r_lever = st.number_input("Lever arm radius [m]", value=0.05)
    A_cm2 = st.number_input("Seal area [cm²]", value=28.0)

    # ---------------- Film Properties ----------------
    st.header("Film Properties")
    d_micro = st.slider("Film thickness per ply [µm]", 20, 100, 35)
    n_plies = st.number_input("Number of plies", value=4)
    rho = st.number_input("Density ρ [kg/m³]", value=920.0)
    cp = st.number_input("Specific heat cp [J/kg·K]", value=2300.0)
    k = st.number_input("Thermal conductivity k [W/m·K]", value=0.33)
    Lf = st.number_input("Latent heat of fusion Lf [J/kg]", value=290000.0)
    dT_seal = st.number_input("ΔT for sealing [K]", value=40.0)

    # ---------------- Contact Conductance ----------------
    st.header("Contact Conductance")
    h0 = st.number_input("Base conductance h0 [W/m²·K]", value=50.0)
    alpha_p = st.number_input("Pressure sensitivity αp [W/m²·K·Pa]", value=0.007)

    # ---------------- Heat Loss Parameters ----------------
    st.header("Heat Loss to Air")
    h_air = st.number_input("Convective coefficient h_air [W/m²·K]", value=10.0)
    eps = st.number_input("Jaw emissivity ε [-]", value=0.8)

    # ---------------- Corrections ----------------
    st.header("Correction Factors")
    delta_t = st.number_input("Thickness variation δt [-]", value=0.1)
    Rq = st.number_input("Surface roughness factor Rq (0 smooth, 0.3 PTFE)", value=0.2)
    h_int = st.number_input("Interface conductance between plies h_int [W/m²·K]", value=200.0)
    include_jaw_mass = st.checkbox("Include jaw finite heat capacity correction", value=True)
    m_jaw = st.number_input("Jaw mass [kg]", value=5.0)
    cp_jaw = st.number_input("Jaw cp [J/kg·K]", value=500.0)

    # ---------------- Calculation ----------------
    t_min = calculate_tmin(T_jaw_C, T0_C, torque, r_lever, A_cm2, d_micro, n_plies,
                           rho, cp, k, Lf, dT_seal, h0, alpha_p,
                           h_air, eps, delta_t, Rq, h_int,
                           include_jaw_mass, m_jaw, cp_jaw)

    st.success(f"👉 Minimum Seal Time = **{t_min:.6f} s**")

    # ---------------- Graphs ----------------
    st.header("Interactive Graphs")

    # Torque curve
    torques = [i for i in range(20, 2001, 50)]
    tmins_torque = [calculate_tmin(T_jaw_C, T0_C, T, r_lever, A_cm2, d_micro, n_plies,
                                   rho, cp, k, Lf, dT_seal, h0, alpha_p,
                                   h_air, eps, delta_t, Rq, h_int,
                                   include_jaw_mass, m_jaw, cp_jaw) for T in torques]

    # Thickness curve
    thicknesses = [i for i in range(20, 101, 5)]
    tmins_thickness = [calculate_tmin(T_jaw_C, T0_C, torque, r_lever, A_cm2, d, n_plies,
                                      rho, cp, k, Lf, dT_seal, h0, alpha_p,
                                      h_air, eps, delta_t, Rq, h_int,
                                      include_jaw_mass, m_jaw, cp_jaw) for d in thicknesses]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(torques, tmins_torque, marker="o")
    ax1.axvline(torque, color="red", linestyle="--", label=f"Selected Torque = {torque}")
    ax1.set_title("Torque vs Minimum Seal Time")
    ax1.set_xlabel("Torque [N·m]")
    ax1.set_ylabel("Seal Time [s]")
    ax1.legend()

    ax2.plot(thicknesses, tmins_thickness, marker="o", color="orange")
    ax2.axvline(d_micro, color="red", linestyle="--", label=f"Selected Thickness = {d_micro} µm")
    ax2.set_title("Film Thickness vs Minimum Seal Time")
    ax2.set_xlabel("Thickness [µm]")
    ax2.set_ylabel("Seal Time [s]")
    ax2.legend()

    st.pyplot(fig)


if __name__ == "__main__":
    main()
