* Parametric 3-Stage Peng Transconductance-Compensated Feedforward (TCFC) Op-Amp Template
* Topology: PMOS folded-cascode + cascoded bias chain + dual-cap TCFC compensation
* Note: device numbering preserves original schematic numbering — 32 MOSFETs total,
*       with non-sequential indices (M0..M9, M11..M23, M57..M58, M61..M66, M69..M70)
*
* Stage 1 (folded-cascode):
*   PMOS diff pair:           M8 (vinn), M9 (vinp), bulk = source = net31
*   Tail current:             M4 (8× BIASCM_P bias mirror at net31)
*   NMOS bottom (8×):         M19 (left, drain=dm_2), M20 (right, drain=net063)
*   NMOS cascode tops (8×):   M15 (left, drain=voutn), M16 (right, drain=voutp)
*   PMOS folded mirror (4×):  M5 (diode at voutn), M6 (mirror voutn → voutp)
*   Output: voutp  (drives Stage 2 & 3 PMOS gates)
*
* Stage 2: PMOS M10 (gate=voutp) common-source
*          NMOS active load: M70 (diode), M69 (mirror to net10) — LOAD2 group
*          Output: net043 → net10 via M69 mirror
*
* PMOS cascoded path (TCFC):  M7 (48× bias from VB1) → net049
*          M66 (48× cascode, gate=VB2) bridges net049 → net10
*          Provides feedforward current path that complements the M69 mirror
*
* Stage 3: Push-pull at vout
*          PMOS M11 (gmf, gate=voutp) + NMOS M23 (gm3, gate=net10)
*
* Cascoded PMOS bias chain (TCFC distinguishing feature):
*   M65 (1×) PMOS diode-connected at VB2 (sets cascode upper bias)
*   M63 (4×) NMOS cascode + M64 (4×) NMOS bottom — sinks current from VB2
*   Each "leaf" PMOS bias mirror (M0-M3) gets a cascode partner (M57-M62):
*     M0  + M57  → VB1   (cascoded reference)
*     M1  + M58  → VB4   (cascoded mirror)
*     M2  + M61  → net7  (gmb intermediate)
*     M3  + M62  → VB3   (cascoded mirror)
*   This gives much higher PMOS bias-mirror output impedance than fan_smc.
*
* Compensation (dual-cap TCFC):
*   C0:  voutp → vout      (Miller cap, ref = 1.1 pF)
*   C1:  net049 → vout     (feedforward cap from PMOS bias node, ref = 1.0 pF)
*
* Bias: I0 → M0/M57 cascode → VB1 ref → mirrors M1-M3, M4 (tail), M5-M7 (folded/cascode)
*       M14 (NMOS diode at VB3) → cascode chain
*       M65 (PMOS diode at VB2) → upper PMOS cascode bias
*
* Parameters (group-based):
*   BIASCM_P: {W_BIASCM_P}, {L_BIASCM_P}, {M_BIASCM_P}    (M0-M7, M57-M58, M61-M62, M65-M66; multipliers 1×, 4×, 8×, 48×)
*   GM1:      {W_GM1}, {L_GM1}, {M_GM1}                    (M8-M9, diff pair)
*   GM2:      {W_GM2}, {L_GM2}, {M_GM2}                    (M10, stage 2 PMOS)
*   GMF:      {W_GMF}, {L_GMF}, {M_GMF}                    (M11, stage 3 PMOS)
*   BIASCM_N: {W_BIASCM_N}, {L_BIASCM_N}, {M_BIASCM_N}    (M12-M20, M63-M64; multipliers 1×, 4×, 8×)
*   LOAD2:    {W_LOAD2}, {L_LOAD2}, {M_LOAD2}              (M69-M70, NMOS mirror at stage-2 output)
*   GM3:      {W_GM3}, {L_GM3}, {M_GM3}                    (M23, stage 3 NMOS)
*   {I_BIAS}, {C_COMP}, {C_COMP2}, {R_IN}, {R_F}, {VDD}, {VIN_P}, {VIN_N}

.title 3-Stage Peng TCFC Operational Amplifier - AC Analysis

* Include SKY130 models
.param mc_mm_switch=0

* Define missing mismatch parameters
.param sky130_fd_pr__nfet_01v8__toxe_slope=0
.param sky130_fd_pr__nfet_01v8__toxe_slope_spectre=0
.param sky130_fd_pr__nfet_01v8__vth0_slope=0
.param sky130_fd_pr__nfet_01v8__vth0_slope1=0
.param sky130_fd_pr__nfet_01v8__vth0_slope_spectre=0
.param sky130_fd_pr__nfet_01v8__voff_slope=0
.param sky130_fd_pr__nfet_01v8__voff_slope_spectre=0
.param sky130_fd_pr__nfet_01v8__nfactor_slope=0
.param sky130_fd_pr__pfet_01v8__toxe_slope=0
.param sky130_fd_pr__pfet_01v8__toxe_slope1=0
.param sky130_fd_pr__pfet_01v8__toxe_slope_spectre=0
.param sky130_fd_pr__pfet_01v8__vth0_slope=0
.param sky130_fd_pr__pfet_01v8__vth0_slope1=0
.param sky130_fd_pr__pfet_01v8__vth0_slope_spectre=0
.param sky130_fd_pr__pfet_01v8__voff_slope=0
.param sky130_fd_pr__pfet_01v8__voff_slope1=0
.param sky130_fd_pr__pfet_01v8__voff_slope_spectre=0
.param sky130_fd_pr__pfet_01v8__nfactor_slope=0
.param sky130_fd_pr__pfet_01v8__nfactor_slope1=0

* Define diff parameters (process variation)
.param sky130_fd_pr__pfet_01v8__wlod_diff=0
.param sky130_fd_pr__pfet_01v8__kvth0_diff=0
.param sky130_fd_pr__pfet_01v8__lkvth0_diff=0
.param sky130_fd_pr__pfet_01v8__wkvth0_diff=0
.param sky130_fd_pr__pfet_01v8__ku0_diff=0
.param sky130_fd_pr__pfet_01v8__lku0_diff=0
.param sky130_fd_pr__pfet_01v8__wku0_diff=0
.param sky130_fd_pr__pfet_01v8__kvsat_diff=0

.include "/home/kevin/tech/sky130/libraries/sky130_fd_pr/latest/cells/pfet_01v8/sky130_fd_pr__pfet_01v8__tt.corner.spice"
.include "/home/kevin/tech/sky130/libraries/sky130_fd_pr/latest/cells/nfet_01v8/sky130_fd_pr__nfet_01v8__tt.corner.spice"

* Power supplies
Vdd vdda 0 DC {VDD}
Vss gnda 0 DC 0

* Input configuration (open-loop)
Vcm_ref  vinp   0    DC {VIN_P}
Vinn     vinn   0    DC {VIN_N} AC 1

* =========================
* PMOS BIAS — cascoded reference + mirrors
* =========================

* M65: PMOS diode-connected at VB2 (cascode upper bias)
Xm65 vb2 vb2 vdda vdda sky130_fd_pr__pfet_01v8 W={W_BIASCM_P} L={L_BIASCM_P} M={M_BIASCM_P}

* M0: PMOS bias bottom (4×) — drain=net2, gate=VB1
Xm0 net2 vb1 vdda vdda sky130_fd_pr__pfet_01v8 W={W_BIASCM_P} L={L_BIASCM_P} M='4*{M_BIASCM_P}'

* M57: PMOS cascode (4×) on top of M0 — generates VB1 reference
Xm57 vb1 vb2 net2 vdda sky130_fd_pr__pfet_01v8 W={W_BIASCM_P} L={L_BIASCM_P} M='4*{M_BIASCM_P}'

* M1 + M58: cascoded mirror -> VB4
Xm1  net3 vb1 vdda vdda sky130_fd_pr__pfet_01v8 W={W_BIASCM_P} L={L_BIASCM_P} M='4*{M_BIASCM_P}'
Xm58 vb4  vb2 net3 vdda sky130_fd_pr__pfet_01v8 W={W_BIASCM_P} L={L_BIASCM_P} M='4*{M_BIASCM_P}'

* M2 + M61: cascoded mirror -> net7 (intermediate)
Xm2  net6 vb1 vdda vdda sky130_fd_pr__pfet_01v8 W={W_BIASCM_P} L={L_BIASCM_P} M='4*{M_BIASCM_P}'
Xm61 net7 vb2 net6 vdda sky130_fd_pr__pfet_01v8 W={W_BIASCM_P} L={L_BIASCM_P} M='4*{M_BIASCM_P}'

* M3 + M62: cascoded mirror -> VB3
Xm3  net8 vb1 vdda vdda sky130_fd_pr__pfet_01v8 W={W_BIASCM_P} L={L_BIASCM_P} M='4*{M_BIASCM_P}'
Xm62 vb3  vb2 net8 vdda sky130_fd_pr__pfet_01v8 W={W_BIASCM_P} L={L_BIASCM_P} M='4*{M_BIASCM_P}'

* M4: tail current for diff pair (8× multiplier)
Xm4 net31 vb1 vdda vdda sky130_fd_pr__pfet_01v8 W={W_BIASCM_P} L={L_BIASCM_P} M='8*{M_BIASCM_P}'

* =========================
* PMOS FOLDED MIRROR (M5, M6)
* =========================

* M5: PMOS diode at VOUTN (4× — folded-cascode top reference)
Xm5 voutn voutn vdda vdda sky130_fd_pr__pfet_01v8 W={W_BIASCM_P} L={L_BIASCM_P} M='4*{M_BIASCM_P}'

* M6: PMOS mirror VOUTN -> VOUTP (4× — Stage 1 output)
Xm6 voutp voutn vdda vdda sky130_fd_pr__pfet_01v8 W={W_BIASCM_P} L={L_BIASCM_P} M='4*{M_BIASCM_P}'

* =========================
* PMOS BIAS to net049 + cascode bridge to net10 (TCFC feedforward path)
* =========================

* M7: PMOS bias mirror -> net049 (48× large multiplier)
Xm7 net049 vb1 vdda vdda sky130_fd_pr__pfet_01v8 W={W_BIASCM_P} L={L_BIASCM_P} M='48*{M_BIASCM_P}'

* M66: PMOS cascode (48×) bridging net049 -> net10 (gate=VB2)
Xm66 net10 vb2 net049 vdda sky130_fd_pr__pfet_01v8 W={W_BIASCM_P} L={L_BIASCM_P} M='48*{M_BIASCM_P}'

* =========================
* STAGE 1: PMOS DIFF PAIR
* =========================

* M8: inverting input (VINN), bulk shorted to source (net31)
Xm8 dm_2 vinn net31 net31 sky130_fd_pr__pfet_01v8 W={W_GM1} L={L_GM1} M={M_GM1}

* M9: non-inverting input (VINP), bulk shorted to source
Xm9 net063 vinp net31 net31 sky130_fd_pr__pfet_01v8 W={W_GM1} L={L_GM1} M={M_GM1}

* =========================
* STAGE 1: NMOS BOTTOM CURRENT SOURCES (8×)
* =========================

* M19: left branch bottom
Xm19 dm_2 vb4 gnda gnda sky130_fd_pr__nfet_01v8 W={W_BIASCM_N} L={L_BIASCM_N} M='8*{M_BIASCM_N}'

* M20: right branch bottom
Xm20 net063 vb4 gnda gnda sky130_fd_pr__nfet_01v8 W={W_BIASCM_N} L={L_BIASCM_N} M='8*{M_BIASCM_N}'

* =========================
* STAGE 1: NMOS CASCODES (8×)
* =========================

* M15: left cascode -> VOUTN
Xm15 voutn vb3 dm_2 gnda sky130_fd_pr__nfet_01v8 W={W_BIASCM_N} L={L_BIASCM_N} M='8*{M_BIASCM_N}'

* M16: right cascode -> VOUTP (Stage 1 output)
Xm16 voutp vb3 net063 gnda sky130_fd_pr__nfet_01v8 W={W_BIASCM_N} L={L_BIASCM_N} M='8*{M_BIASCM_N}'

* =========================
* NMOS BIAS GENERATION
* =========================

* M14: NMOS diode-connected at VB3 (1×)
Xm14 vb3 vb3 gnda gnda sky130_fd_pr__nfet_01v8 W={W_BIASCM_N} L={L_BIASCM_N} M={M_BIASCM_N}

* M17 + M12: VB4 cascode generation (4×)
Xm17 net54 vb4 gnda gnda sky130_fd_pr__nfet_01v8 W={W_BIASCM_N} L={L_BIASCM_N} M='4*{M_BIASCM_N}'
Xm12 vb4  vb3 net54 gnda sky130_fd_pr__nfet_01v8 W={W_BIASCM_N} L={L_BIASCM_N} M='4*{M_BIASCM_N}'

* M18 + M13: net7 cascode chain to ground (4×)
Xm18 net56 vb4 gnda gnda sky130_fd_pr__nfet_01v8 W={W_BIASCM_N} L={L_BIASCM_N} M='4*{M_BIASCM_N}'
Xm13 net7  vb3 net56 gnda sky130_fd_pr__nfet_01v8 W={W_BIASCM_N} L={L_BIASCM_N} M='4*{M_BIASCM_N}'

* M64 + M63: VB2 cascode generation chain (4×)
Xm64 net9 vb4 gnda gnda sky130_fd_pr__nfet_01v8 W={W_BIASCM_N} L={L_BIASCM_N} M='4*{M_BIASCM_N}'
Xm63 vb2  vb3 net9 gnda sky130_fd_pr__nfet_01v8 W={W_BIASCM_N} L={L_BIASCM_N} M='4*{M_BIASCM_N}'

* =========================
* STAGE 2: COMMON-SOURCE
* =========================

* M10: PMOS gain stage (gate=VOUTP, drain=net043)
Xm10 net043 voutp vdda vdda sky130_fd_pr__pfet_01v8 W={W_GM2} L={L_GM2} M={M_GM2}

* M70: NMOS diode-connected at net043 (LOAD2 group, separate from BIASCM_N)
Xm70 net043 net043 gnda gnda sky130_fd_pr__nfet_01v8 W={W_LOAD2} L={L_LOAD2} M={M_LOAD2}

* M69: NMOS mirror -> net10 (drives Stage 3 NMOS gate)
Xm69 net10 net043 gnda gnda sky130_fd_pr__nfet_01v8 W={W_LOAD2} L={L_LOAD2} M={M_LOAD2}

* =========================
* STAGE 3: PUSH-PULL OUTPUT
* =========================

* M11: PMOS output (gate=VOUTP)
Xm11 vout voutp vdda vdda sky130_fd_pr__pfet_01v8 W={W_GMF} L={L_GMF} M={M_GMF}

* M23: NMOS output (gate=net10)
Xm23 vout net10 gnda gnda sky130_fd_pr__nfet_01v8 W={W_GM3} L={L_GM3} M={M_GM3}

* =========================
* BIAS CURRENT SOURCE
* =========================

I0 vb1 gnda DC {I_BIAS}

* =========================
* COMPENSATION NETWORK (dual-cap TCFC)
* =========================

* C0: Miller cap from Stage 1 output to final output
C0 voutp vout {C_COMP}

* C1: feedforward cap from PMOS bias node net049 to final output
C1 net049 vout {C_COMP2}

* =========================
* OUTPUT LOAD
* =========================

Cload vout 0 10p

* =========================
* SAVE DEVICE CURRENTS
* =========================

.save all
* MOSFET drain currents (PMOS)
.save @m.xm0.msky130_fd_pr__pfet_01v8[id]
.save @m.xm1.msky130_fd_pr__pfet_01v8[id]
.save @m.xm2.msky130_fd_pr__pfet_01v8[id]
.save @m.xm3.msky130_fd_pr__pfet_01v8[id]
.save @m.xm4.msky130_fd_pr__pfet_01v8[id]
.save @m.xm5.msky130_fd_pr__pfet_01v8[id]
.save @m.xm6.msky130_fd_pr__pfet_01v8[id]
.save @m.xm7.msky130_fd_pr__pfet_01v8[id]
.save @m.xm8.msky130_fd_pr__pfet_01v8[id]
.save @m.xm9.msky130_fd_pr__pfet_01v8[id]
.save @m.xm10.msky130_fd_pr__pfet_01v8[id]
.save @m.xm11.msky130_fd_pr__pfet_01v8[id]
.save @m.xm57.msky130_fd_pr__pfet_01v8[id]
.save @m.xm58.msky130_fd_pr__pfet_01v8[id]
.save @m.xm61.msky130_fd_pr__pfet_01v8[id]
.save @m.xm62.msky130_fd_pr__pfet_01v8[id]
.save @m.xm65.msky130_fd_pr__pfet_01v8[id]
.save @m.xm66.msky130_fd_pr__pfet_01v8[id]
* MOSFET drain currents (NMOS)
.save @m.xm12.msky130_fd_pr__nfet_01v8[id]
.save @m.xm13.msky130_fd_pr__nfet_01v8[id]
.save @m.xm14.msky130_fd_pr__nfet_01v8[id]
.save @m.xm15.msky130_fd_pr__nfet_01v8[id]
.save @m.xm16.msky130_fd_pr__nfet_01v8[id]
.save @m.xm17.msky130_fd_pr__nfet_01v8[id]
.save @m.xm18.msky130_fd_pr__nfet_01v8[id]
.save @m.xm19.msky130_fd_pr__nfet_01v8[id]
.save @m.xm20.msky130_fd_pr__nfet_01v8[id]
.save @m.xm23.msky130_fd_pr__nfet_01v8[id]
.save @m.xm63.msky130_fd_pr__nfet_01v8[id]
.save @m.xm64.msky130_fd_pr__nfet_01v8[id]
.save @m.xm69.msky130_fd_pr__nfet_01v8[id]
.save @m.xm70.msky130_fd_pr__nfet_01v8[id]
* MOSFET source currents (PMOS)
.save @m.xm0.msky130_fd_pr__pfet_01v8[is]
.save @m.xm1.msky130_fd_pr__pfet_01v8[is]
.save @m.xm2.msky130_fd_pr__pfet_01v8[is]
.save @m.xm3.msky130_fd_pr__pfet_01v8[is]
.save @m.xm4.msky130_fd_pr__pfet_01v8[is]
.save @m.xm5.msky130_fd_pr__pfet_01v8[is]
.save @m.xm6.msky130_fd_pr__pfet_01v8[is]
.save @m.xm7.msky130_fd_pr__pfet_01v8[is]
.save @m.xm8.msky130_fd_pr__pfet_01v8[is]
.save @m.xm9.msky130_fd_pr__pfet_01v8[is]
.save @m.xm10.msky130_fd_pr__pfet_01v8[is]
.save @m.xm11.msky130_fd_pr__pfet_01v8[is]
.save @m.xm57.msky130_fd_pr__pfet_01v8[is]
.save @m.xm58.msky130_fd_pr__pfet_01v8[is]
.save @m.xm61.msky130_fd_pr__pfet_01v8[is]
.save @m.xm62.msky130_fd_pr__pfet_01v8[is]
.save @m.xm65.msky130_fd_pr__pfet_01v8[is]
.save @m.xm66.msky130_fd_pr__pfet_01v8[is]
* MOSFET source currents (NMOS)
.save @m.xm12.msky130_fd_pr__nfet_01v8[is]
.save @m.xm13.msky130_fd_pr__nfet_01v8[is]
.save @m.xm14.msky130_fd_pr__nfet_01v8[is]
.save @m.xm15.msky130_fd_pr__nfet_01v8[is]
.save @m.xm16.msky130_fd_pr__nfet_01v8[is]
.save @m.xm17.msky130_fd_pr__nfet_01v8[is]
.save @m.xm18.msky130_fd_pr__nfet_01v8[is]
.save @m.xm19.msky130_fd_pr__nfet_01v8[is]
.save @m.xm20.msky130_fd_pr__nfet_01v8[is]
.save @m.xm23.msky130_fd_pr__nfet_01v8[is]
.save @m.xm63.msky130_fd_pr__nfet_01v8[is]
.save @m.xm64.msky130_fd_pr__nfet_01v8[is]
.save @m.xm69.msky130_fd_pr__nfet_01v8[is]
.save @m.xm70.msky130_fd_pr__nfet_01v8[is]
* MOSFET gate currents (PMOS)
.save @m.xm0.msky130_fd_pr__pfet_01v8[ig]
.save @m.xm1.msky130_fd_pr__pfet_01v8[ig]
.save @m.xm2.msky130_fd_pr__pfet_01v8[ig]
.save @m.xm3.msky130_fd_pr__pfet_01v8[ig]
.save @m.xm4.msky130_fd_pr__pfet_01v8[ig]
.save @m.xm5.msky130_fd_pr__pfet_01v8[ig]
.save @m.xm6.msky130_fd_pr__pfet_01v8[ig]
.save @m.xm7.msky130_fd_pr__pfet_01v8[ig]
.save @m.xm8.msky130_fd_pr__pfet_01v8[ig]
.save @m.xm9.msky130_fd_pr__pfet_01v8[ig]
.save @m.xm10.msky130_fd_pr__pfet_01v8[ig]
.save @m.xm11.msky130_fd_pr__pfet_01v8[ig]
.save @m.xm57.msky130_fd_pr__pfet_01v8[ig]
.save @m.xm58.msky130_fd_pr__pfet_01v8[ig]
.save @m.xm61.msky130_fd_pr__pfet_01v8[ig]
.save @m.xm62.msky130_fd_pr__pfet_01v8[ig]
.save @m.xm65.msky130_fd_pr__pfet_01v8[ig]
.save @m.xm66.msky130_fd_pr__pfet_01v8[ig]
* MOSFET gate currents (NMOS)
.save @m.xm12.msky130_fd_pr__nfet_01v8[ig]
.save @m.xm13.msky130_fd_pr__nfet_01v8[ig]
.save @m.xm14.msky130_fd_pr__nfet_01v8[ig]
.save @m.xm15.msky130_fd_pr__nfet_01v8[ig]
.save @m.xm16.msky130_fd_pr__nfet_01v8[ig]
.save @m.xm17.msky130_fd_pr__nfet_01v8[ig]
.save @m.xm18.msky130_fd_pr__nfet_01v8[ig]
.save @m.xm19.msky130_fd_pr__nfet_01v8[ig]
.save @m.xm20.msky130_fd_pr__nfet_01v8[ig]
.save @m.xm23.msky130_fd_pr__nfet_01v8[ig]
.save @m.xm63.msky130_fd_pr__nfet_01v8[ig]
.save @m.xm64.msky130_fd_pr__nfet_01v8[ig]
.save @m.xm69.msky130_fd_pr__nfet_01v8[ig]
.save @m.xm70.msky130_fd_pr__nfet_01v8[ig]
* MOSFET bulk currents (PMOS)
.save @m.xm0.msky130_fd_pr__pfet_01v8[ib]
.save @m.xm1.msky130_fd_pr__pfet_01v8[ib]
.save @m.xm2.msky130_fd_pr__pfet_01v8[ib]
.save @m.xm3.msky130_fd_pr__pfet_01v8[ib]
.save @m.xm4.msky130_fd_pr__pfet_01v8[ib]
.save @m.xm5.msky130_fd_pr__pfet_01v8[ib]
.save @m.xm6.msky130_fd_pr__pfet_01v8[ib]
.save @m.xm7.msky130_fd_pr__pfet_01v8[ib]
.save @m.xm8.msky130_fd_pr__pfet_01v8[ib]
.save @m.xm9.msky130_fd_pr__pfet_01v8[ib]
.save @m.xm10.msky130_fd_pr__pfet_01v8[ib]
.save @m.xm11.msky130_fd_pr__pfet_01v8[ib]
.save @m.xm57.msky130_fd_pr__pfet_01v8[ib]
.save @m.xm58.msky130_fd_pr__pfet_01v8[ib]
.save @m.xm61.msky130_fd_pr__pfet_01v8[ib]
.save @m.xm62.msky130_fd_pr__pfet_01v8[ib]
.save @m.xm65.msky130_fd_pr__pfet_01v8[ib]
.save @m.xm66.msky130_fd_pr__pfet_01v8[ib]
* MOSFET bulk currents (NMOS)
.save @m.xm12.msky130_fd_pr__nfet_01v8[ib]
.save @m.xm13.msky130_fd_pr__nfet_01v8[ib]
.save @m.xm14.msky130_fd_pr__nfet_01v8[ib]
.save @m.xm15.msky130_fd_pr__nfet_01v8[ib]
.save @m.xm16.msky130_fd_pr__nfet_01v8[ib]
.save @m.xm17.msky130_fd_pr__nfet_01v8[ib]
.save @m.xm18.msky130_fd_pr__nfet_01v8[ib]
.save @m.xm19.msky130_fd_pr__nfet_01v8[ib]
.save @m.xm20.msky130_fd_pr__nfet_01v8[ib]
.save @m.xm23.msky130_fd_pr__nfet_01v8[ib]
.save @m.xm63.msky130_fd_pr__nfet_01v8[ib]
.save @m.xm64.msky130_fd_pr__nfet_01v8[ib]
.save @m.xm69.msky130_fd_pr__nfet_01v8[ib]
.save @m.xm70.msky130_fd_pr__nfet_01v8[ib]
* MOSFET operating point parameters (PMOS)
.save @m.xm0.msky130_fd_pr__pfet_01v8[vgs] @m.xm0.msky130_fd_pr__pfet_01v8[vds] @m.xm0.msky130_fd_pr__pfet_01v8[vdsat] @m.xm0.msky130_fd_pr__pfet_01v8[vth]
.save @m.xm1.msky130_fd_pr__pfet_01v8[vgs] @m.xm1.msky130_fd_pr__pfet_01v8[vds] @m.xm1.msky130_fd_pr__pfet_01v8[vdsat] @m.xm1.msky130_fd_pr__pfet_01v8[vth]
.save @m.xm2.msky130_fd_pr__pfet_01v8[vgs] @m.xm2.msky130_fd_pr__pfet_01v8[vds] @m.xm2.msky130_fd_pr__pfet_01v8[vdsat] @m.xm2.msky130_fd_pr__pfet_01v8[vth]
.save @m.xm3.msky130_fd_pr__pfet_01v8[vgs] @m.xm3.msky130_fd_pr__pfet_01v8[vds] @m.xm3.msky130_fd_pr__pfet_01v8[vdsat] @m.xm3.msky130_fd_pr__pfet_01v8[vth]
.save @m.xm4.msky130_fd_pr__pfet_01v8[vgs] @m.xm4.msky130_fd_pr__pfet_01v8[vds] @m.xm4.msky130_fd_pr__pfet_01v8[vdsat] @m.xm4.msky130_fd_pr__pfet_01v8[vth]
.save @m.xm5.msky130_fd_pr__pfet_01v8[vgs] @m.xm5.msky130_fd_pr__pfet_01v8[vds] @m.xm5.msky130_fd_pr__pfet_01v8[vdsat] @m.xm5.msky130_fd_pr__pfet_01v8[vth]
.save @m.xm6.msky130_fd_pr__pfet_01v8[vgs] @m.xm6.msky130_fd_pr__pfet_01v8[vds] @m.xm6.msky130_fd_pr__pfet_01v8[vdsat] @m.xm6.msky130_fd_pr__pfet_01v8[vth]
.save @m.xm7.msky130_fd_pr__pfet_01v8[vgs] @m.xm7.msky130_fd_pr__pfet_01v8[vds] @m.xm7.msky130_fd_pr__pfet_01v8[vdsat] @m.xm7.msky130_fd_pr__pfet_01v8[vth]
.save @m.xm8.msky130_fd_pr__pfet_01v8[vgs] @m.xm8.msky130_fd_pr__pfet_01v8[vds] @m.xm8.msky130_fd_pr__pfet_01v8[vdsat] @m.xm8.msky130_fd_pr__pfet_01v8[vth]
.save @m.xm9.msky130_fd_pr__pfet_01v8[vgs] @m.xm9.msky130_fd_pr__pfet_01v8[vds] @m.xm9.msky130_fd_pr__pfet_01v8[vdsat] @m.xm9.msky130_fd_pr__pfet_01v8[vth]
.save @m.xm10.msky130_fd_pr__pfet_01v8[vgs] @m.xm10.msky130_fd_pr__pfet_01v8[vds] @m.xm10.msky130_fd_pr__pfet_01v8[vdsat] @m.xm10.msky130_fd_pr__pfet_01v8[vth]
.save @m.xm11.msky130_fd_pr__pfet_01v8[vgs] @m.xm11.msky130_fd_pr__pfet_01v8[vds] @m.xm11.msky130_fd_pr__pfet_01v8[vdsat] @m.xm11.msky130_fd_pr__pfet_01v8[vth]
.save @m.xm57.msky130_fd_pr__pfet_01v8[vgs] @m.xm57.msky130_fd_pr__pfet_01v8[vds] @m.xm57.msky130_fd_pr__pfet_01v8[vdsat] @m.xm57.msky130_fd_pr__pfet_01v8[vth]
.save @m.xm58.msky130_fd_pr__pfet_01v8[vgs] @m.xm58.msky130_fd_pr__pfet_01v8[vds] @m.xm58.msky130_fd_pr__pfet_01v8[vdsat] @m.xm58.msky130_fd_pr__pfet_01v8[vth]
.save @m.xm61.msky130_fd_pr__pfet_01v8[vgs] @m.xm61.msky130_fd_pr__pfet_01v8[vds] @m.xm61.msky130_fd_pr__pfet_01v8[vdsat] @m.xm61.msky130_fd_pr__pfet_01v8[vth]
.save @m.xm62.msky130_fd_pr__pfet_01v8[vgs] @m.xm62.msky130_fd_pr__pfet_01v8[vds] @m.xm62.msky130_fd_pr__pfet_01v8[vdsat] @m.xm62.msky130_fd_pr__pfet_01v8[vth]
.save @m.xm65.msky130_fd_pr__pfet_01v8[vgs] @m.xm65.msky130_fd_pr__pfet_01v8[vds] @m.xm65.msky130_fd_pr__pfet_01v8[vdsat] @m.xm65.msky130_fd_pr__pfet_01v8[vth]
.save @m.xm66.msky130_fd_pr__pfet_01v8[vgs] @m.xm66.msky130_fd_pr__pfet_01v8[vds] @m.xm66.msky130_fd_pr__pfet_01v8[vdsat] @m.xm66.msky130_fd_pr__pfet_01v8[vth]
* MOSFET operating point parameters (NMOS)
.save @m.xm12.msky130_fd_pr__nfet_01v8[vgs] @m.xm12.msky130_fd_pr__nfet_01v8[vds] @m.xm12.msky130_fd_pr__nfet_01v8[vdsat] @m.xm12.msky130_fd_pr__nfet_01v8[vth]
.save @m.xm13.msky130_fd_pr__nfet_01v8[vgs] @m.xm13.msky130_fd_pr__nfet_01v8[vds] @m.xm13.msky130_fd_pr__nfet_01v8[vdsat] @m.xm13.msky130_fd_pr__nfet_01v8[vth]
.save @m.xm14.msky130_fd_pr__nfet_01v8[vgs] @m.xm14.msky130_fd_pr__nfet_01v8[vds] @m.xm14.msky130_fd_pr__nfet_01v8[vdsat] @m.xm14.msky130_fd_pr__nfet_01v8[vth]
.save @m.xm15.msky130_fd_pr__nfet_01v8[vgs] @m.xm15.msky130_fd_pr__nfet_01v8[vds] @m.xm15.msky130_fd_pr__nfet_01v8[vdsat] @m.xm15.msky130_fd_pr__nfet_01v8[vth]
.save @m.xm16.msky130_fd_pr__nfet_01v8[vgs] @m.xm16.msky130_fd_pr__nfet_01v8[vds] @m.xm16.msky130_fd_pr__nfet_01v8[vdsat] @m.xm16.msky130_fd_pr__nfet_01v8[vth]
.save @m.xm17.msky130_fd_pr__nfet_01v8[vgs] @m.xm17.msky130_fd_pr__nfet_01v8[vds] @m.xm17.msky130_fd_pr__nfet_01v8[vdsat] @m.xm17.msky130_fd_pr__nfet_01v8[vth]
.save @m.xm18.msky130_fd_pr__nfet_01v8[vgs] @m.xm18.msky130_fd_pr__nfet_01v8[vds] @m.xm18.msky130_fd_pr__nfet_01v8[vdsat] @m.xm18.msky130_fd_pr__nfet_01v8[vth]
.save @m.xm19.msky130_fd_pr__nfet_01v8[vgs] @m.xm19.msky130_fd_pr__nfet_01v8[vds] @m.xm19.msky130_fd_pr__nfet_01v8[vdsat] @m.xm19.msky130_fd_pr__nfet_01v8[vth]
.save @m.xm20.msky130_fd_pr__nfet_01v8[vgs] @m.xm20.msky130_fd_pr__nfet_01v8[vds] @m.xm20.msky130_fd_pr__nfet_01v8[vdsat] @m.xm20.msky130_fd_pr__nfet_01v8[vth]
.save @m.xm23.msky130_fd_pr__nfet_01v8[vgs] @m.xm23.msky130_fd_pr__nfet_01v8[vds] @m.xm23.msky130_fd_pr__nfet_01v8[vdsat] @m.xm23.msky130_fd_pr__nfet_01v8[vth]
.save @m.xm63.msky130_fd_pr__nfet_01v8[vgs] @m.xm63.msky130_fd_pr__nfet_01v8[vds] @m.xm63.msky130_fd_pr__nfet_01v8[vdsat] @m.xm63.msky130_fd_pr__nfet_01v8[vth]
.save @m.xm64.msky130_fd_pr__nfet_01v8[vgs] @m.xm64.msky130_fd_pr__nfet_01v8[vds] @m.xm64.msky130_fd_pr__nfet_01v8[vdsat] @m.xm64.msky130_fd_pr__nfet_01v8[vth]
.save @m.xm69.msky130_fd_pr__nfet_01v8[vgs] @m.xm69.msky130_fd_pr__nfet_01v8[vds] @m.xm69.msky130_fd_pr__nfet_01v8[vdsat] @m.xm69.msky130_fd_pr__nfet_01v8[vth]
.save @m.xm70.msky130_fd_pr__nfet_01v8[vgs] @m.xm70.msky130_fd_pr__nfet_01v8[vds] @m.xm70.msky130_fd_pr__nfet_01v8[vdsat] @m.xm70.msky130_fd_pr__nfet_01v8[vth]
* Source currents
.save @i0[i]


.save @m.xm0.msky130_fd_pr__pfet_01v8[gm] @m.xm0.msky130_fd_pr__pfet_01v8[gds]
.save @m.xm1.msky130_fd_pr__pfet_01v8[gm] @m.xm1.msky130_fd_pr__pfet_01v8[gds]
.save @m.xm2.msky130_fd_pr__pfet_01v8[gm] @m.xm2.msky130_fd_pr__pfet_01v8[gds]
.save @m.xm3.msky130_fd_pr__pfet_01v8[gm] @m.xm3.msky130_fd_pr__pfet_01v8[gds]
.save @m.xm4.msky130_fd_pr__pfet_01v8[gm] @m.xm4.msky130_fd_pr__pfet_01v8[gds]
.save @m.xm5.msky130_fd_pr__pfet_01v8[gm] @m.xm5.msky130_fd_pr__pfet_01v8[gds]
.save @m.xm6.msky130_fd_pr__pfet_01v8[gm] @m.xm6.msky130_fd_pr__pfet_01v8[gds]
.save @m.xm7.msky130_fd_pr__pfet_01v8[gm] @m.xm7.msky130_fd_pr__pfet_01v8[gds]
.save @m.xm8.msky130_fd_pr__pfet_01v8[gm] @m.xm8.msky130_fd_pr__pfet_01v8[gds]
.save @m.xm9.msky130_fd_pr__pfet_01v8[gm] @m.xm9.msky130_fd_pr__pfet_01v8[gds]
.save @m.xm10.msky130_fd_pr__pfet_01v8[gm] @m.xm10.msky130_fd_pr__pfet_01v8[gds]
.save @m.xm11.msky130_fd_pr__pfet_01v8[gm] @m.xm11.msky130_fd_pr__pfet_01v8[gds]
.save @m.xm12.msky130_fd_pr__nfet_01v8[gm] @m.xm12.msky130_fd_pr__nfet_01v8[gds]
.save @m.xm13.msky130_fd_pr__nfet_01v8[gm] @m.xm13.msky130_fd_pr__nfet_01v8[gds]
.save @m.xm14.msky130_fd_pr__nfet_01v8[gm] @m.xm14.msky130_fd_pr__nfet_01v8[gds]
.save @m.xm15.msky130_fd_pr__nfet_01v8[gm] @m.xm15.msky130_fd_pr__nfet_01v8[gds]
.save @m.xm16.msky130_fd_pr__nfet_01v8[gm] @m.xm16.msky130_fd_pr__nfet_01v8[gds]
.save @m.xm17.msky130_fd_pr__nfet_01v8[gm] @m.xm17.msky130_fd_pr__nfet_01v8[gds]
.save @m.xm18.msky130_fd_pr__nfet_01v8[gm] @m.xm18.msky130_fd_pr__nfet_01v8[gds]
.save @m.xm19.msky130_fd_pr__nfet_01v8[gm] @m.xm19.msky130_fd_pr__nfet_01v8[gds]
.save @m.xm20.msky130_fd_pr__nfet_01v8[gm] @m.xm20.msky130_fd_pr__nfet_01v8[gds]
.save @m.xm21.msky130_fd_pr__nfet_01v8[gm] @m.xm21.msky130_fd_pr__nfet_01v8[gds]
.save @m.xm22.msky130_fd_pr__nfet_01v8[gm] @m.xm22.msky130_fd_pr__nfet_01v8[gds]
.save @m.xm23.msky130_fd_pr__nfet_01v8[gm] @m.xm23.msky130_fd_pr__nfet_01v8[gds]

.save @m.xm57.msky130_fd_pr__pfet_01v8[gm] @m.xm57.msky130_fd_pr__pfet_01v8[gds]
.save @m.xm58.msky130_fd_pr__pfet_01v8[gm] @m.xm58.msky130_fd_pr__pfet_01v8[gds]
.save @m.xm61.msky130_fd_pr__pfet_01v8[gm] @m.xm61.msky130_fd_pr__pfet_01v8[gds]
.save @m.xm62.msky130_fd_pr__pfet_01v8[gm] @m.xm62.msky130_fd_pr__pfet_01v8[gds]
.save @m.xm65.msky130_fd_pr__pfet_01v8[gm] @m.xm65.msky130_fd_pr__pfet_01v8[gds]
.save @m.xm66.msky130_fd_pr__pfet_01v8[gm] @m.xm66.msky130_fd_pr__pfet_01v8[gds]
.save @m.xm63.msky130_fd_pr__nfet_01v8[gm] @m.xm63.msky130_fd_pr__nfet_01v8[gds]
.save @m.xm64.msky130_fd_pr__nfet_01v8[gm] @m.xm64.msky130_fd_pr__nfet_01v8[gds]
.save @m.xm69.msky130_fd_pr__nfet_01v8[gm] @m.xm69.msky130_fd_pr__nfet_01v8[gds]
.save @m.xm70.msky130_fd_pr__nfet_01v8[gm] @m.xm70.msky130_fd_pr__nfet_01v8[gds]

* =========================
* ANALYSIS
* =========================

.op
.ac dec 100 1 10G

.end
