* Parametric 3-Stage Leung Nested-Miller Compensation with Nulling Resistor (NMCNR)
* Topology: PMOS folded-cascode + class-A NMOS output + nested Miller comp with R0 nulling
* Note: 24 MOSFETs, same numbering as fan_smc (M0..M23).  Difference vs fan_smc:
*   - M11 is a CLASS-A bias PMOS (gate=net013, 10× multiplier) — NOT signal-driven
*   - Compensation network is nested Miller with nulling resistor (3 passives)
*
* Stage 1 (folded-cascode):
*   PMOS diff pair:  M8 (vinn), M9 (vinp), bulk = source = net31
*   Tail current:    M4 (4× BIASCM_P bias mirror at net31)
*   NMOS bottom (8×): M19 (left), M20 (right)
*   NMOS cascode (4×): M15 (left), M16 (right) → drains voutn, net050
*   PMOS folded mirror: M5 (diode at voutn), M6 (mirror voutn → net050)
*   Output: net050  (drives Stage 2 PMOS gate only — NOT Stage 3 like in fan_smc)
*
* Stage 2: PMOS M10 (gate=net050) common-source
*          NMOS load: M21 (diode), M22 (mirror) → output net049
*
* Stage 3: CLASS-A (not push-pull) at vout
*          PMOS M11 (gate=net013, 10× — STATIC current source)
*          NMOS M23 (gate=net049, gm3 — signal-driven)
*          → only the NMOS provides Stage 3 transconductance
*
* Bias: I0 → M0 (diode PMOS at net013) → mirrors M1-M7
*       M14 (diode NMOS, VB3) + M12 cascode → VB4
*
* Nested Miller Compensation with Nulling Resistor (NMCNR) — distinguishing feature:
*   net044 is an intermediate "Miller-network" node tying:
*     C0 (5 pF ref):  net050  ↔ net044   (Stage 1 output to Miller hub)
*     C1 (3 pF ref):  net044  ↔ net049   (Miller hub to Stage 2 output)
*     R0 (10 kΩ ref): net044  ↔ vout     (nulling resistor to final output)
*   The nulling resistor R0 lets you place a zero in a controlled location to
*   cancel a non-dominant pole — improves phase margin without large caps.
*
* Parameters (group-based):
*   BIASCM_P: {W_BIASCM_P}, {L_BIASCM_P}, {M_BIASCM_P}    (M0-M7, M11; M4=4×, M11=10×)
*   GM1:      {W_GM1}, {L_GM1}, {M_GM1}                    (M8-M9, diff pair)
*   GM2:      {W_GM2}, {L_GM2}, {M_GM2}                    (M10, stage 2 PMOS)
*   BIASCM_N: {W_BIASCM_N}, {L_BIASCM_N}, {M_BIASCM_N}    (M12-M20)
*   LOAD2:    {W_LOAD2}, {L_LOAD2}, {M_LOAD2}              (M21-M22, stage 2 NMOS load)
*   GM3:      {W_GM3}, {L_GM3}, {M_GM3}                    (M23, stage 3 NMOS)
*   {I_BIAS}, {C_COMP}, {C_COMP2}, {R_NULL}, {R_IN}, {R_F}, {VDD}, {VIN_P}, {VIN_N}

.title 3-Stage Leung NMCNR Operational Amplifier - AC Analysis

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
* PMOS BIAS MIRROR (M0-M7)
* =========================

* M0: diode-connected PMOS bias reference
Xm0 net013 net013 vdda vdda sky130_fd_pr__pfet_01v8 W={W_BIASCM_P} L={L_BIASCM_P} M={M_BIASCM_P}

* M1: bias mirror -> VB4 generation
Xm1 vb4 net013 vdda vdda sky130_fd_pr__pfet_01v8 W={W_BIASCM_P} L={L_BIASCM_P} M={M_BIASCM_P}

* M2: bias mirror -> DM_1 (cascode bias branch)
Xm2 dm_1 net013 vdda vdda sky130_fd_pr__pfet_01v8 W={W_BIASCM_P} L={L_BIASCM_P} M={M_BIASCM_P}

* M3: bias mirror -> VB3 generation
Xm3 vb3 net013 vdda vdda sky130_fd_pr__pfet_01v8 W={W_BIASCM_P} L={L_BIASCM_P} M={M_BIASCM_P}

* M4: tail current for diff pair (4× multiplier)
Xm4 net31 net013 vdda vdda sky130_fd_pr__pfet_01v8 W={W_BIASCM_P} L={L_BIASCM_P} M='4*{M_BIASCM_P}'

* M5: diode-connected PMOS at VOUTN (folded-cascode top reference)
Xm5 voutn voutn vdda vdda sky130_fd_pr__pfet_01v8 W={W_BIASCM_P} L={L_BIASCM_P} M={M_BIASCM_P}

* M6: PMOS mirror VOUTN -> net050 (Stage 1 output)
Xm6 net050 voutn vdda vdda sky130_fd_pr__pfet_01v8 W={W_BIASCM_P} L={L_BIASCM_P} M={M_BIASCM_P}

* M7: bias mirror -> net049 (Stage 2 output bias)
Xm7 net049 net013 vdda vdda sky130_fd_pr__pfet_01v8 W={W_BIASCM_P} L={L_BIASCM_P} M={M_BIASCM_P}

* =========================
* STAGE 1: PMOS DIFF PAIR
* =========================

* M8: inverting input (VINN), bulk shorted to source
Xm8 dm_2 vinn net31 net31 sky130_fd_pr__pfet_01v8 W={W_GM1} L={L_GM1} M={M_GM1}

* M9: non-inverting input (VINP), bulk shorted to source
Xm9 net063 vinp net31 net31 sky130_fd_pr__pfet_01v8 W={W_GM1} L={L_GM1} M={M_GM1}

* =========================
* STAGE 1: NMOS CASCODE LOAD
* =========================

* M19: left branch bottom (8×)
Xm19 dm_2 vb4 gnda gnda sky130_fd_pr__nfet_01v8 W={W_BIASCM_N} L={L_BIASCM_N} M='8*{M_BIASCM_N}'

* M20: right branch bottom (8×)
Xm20 net063 vb4 gnda gnda sky130_fd_pr__nfet_01v8 W={W_BIASCM_N} L={L_BIASCM_N} M='8*{M_BIASCM_N}'

* M15: left cascode (4×) -> VOUTN
Xm15 voutn vb3 dm_2 gnda sky130_fd_pr__nfet_01v8 W={W_BIASCM_N} L={L_BIASCM_N} M='4*{M_BIASCM_N}'

* M16: right cascode (4×) -> net050
Xm16 net050 vb3 net063 gnda sky130_fd_pr__nfet_01v8 W={W_BIASCM_N} L={L_BIASCM_N} M='4*{M_BIASCM_N}'

* =========================
* NMOS BIAS GENERATION
* =========================

* M14: NMOS diode-connected -> VB3
Xm14 vb3 vb3 gnda gnda sky130_fd_pr__nfet_01v8 W={W_BIASCM_N} L={L_BIASCM_N} M={M_BIASCM_N}

* M17 + M12: VB4 cascode chain (4×)
Xm17 net54 vb4 gnda gnda sky130_fd_pr__nfet_01v8 W={W_BIASCM_N} L={L_BIASCM_N} M='4*{M_BIASCM_N}'
Xm12 vb4  vb3 net54 gnda sky130_fd_pr__nfet_01v8 W={W_BIASCM_N} L={L_BIASCM_N} M='4*{M_BIASCM_N}'

* M18 + M13: DM_1 cascode chain (4×)
Xm18 net56 vb4 gnda gnda sky130_fd_pr__nfet_01v8 W={W_BIASCM_N} L={L_BIASCM_N} M='4*{M_BIASCM_N}'
Xm13 dm_1 vb3 net56 gnda sky130_fd_pr__nfet_01v8 W={W_BIASCM_N} L={L_BIASCM_N} M='4*{M_BIASCM_N}'

* =========================
* STAGE 2: COMMON-SOURCE
* =========================

* M10: PMOS gain stage (gate=net050)
Xm10 net043 net050 vdda vdda sky130_fd_pr__pfet_01v8 W={W_GM2} L={L_GM2} M={M_GM2}

* M21: NMOS diode-connected load
Xm21 net043 net043 gnda gnda sky130_fd_pr__nfet_01v8 W={W_LOAD2} L={L_LOAD2} M={M_LOAD2}

* M22: NMOS mirror -> net049
Xm22 net049 net043 gnda gnda sky130_fd_pr__nfet_01v8 W={W_LOAD2} L={L_LOAD2} M={M_LOAD2}

* =========================
* STAGE 3: CLASS-A (not push-pull)
* =========================

* M11: PMOS class-A bias source (gate=net013, 10× — STATIC, no signal)
Xm11 vout net013 vdda vdda sky130_fd_pr__pfet_01v8 W={W_BIASCM_P} L={L_BIASCM_P} M='10*{M_BIASCM_P}'

* M23: NMOS output (gate=net049, signal-driven)
Xm23 vout net049 gnda gnda sky130_fd_pr__nfet_01v8 W={W_GM3} L={L_GM3} M={M_GM3}

* =========================
* BIAS CURRENT SOURCE
* =========================

I0 net013 gnda DC {I_BIAS}

* =========================
* COMPENSATION NETWORK (NMCNR — nested Miller with nulling resistor)
* =========================

* C0: Miller cap from Stage 1 output to compensation hub
C0 net050 net044 {C_COMP}

* C1: Miller cap from compensation hub to Stage 2 output
C1 net044 net049 {C_COMP2}

* R0: nulling resistor from compensation hub to final output
R0 net044 vout {R_NULL}

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
.save @m.xm21.msky130_fd_pr__nfet_01v8[id]
.save @m.xm22.msky130_fd_pr__nfet_01v8[id]
.save @m.xm23.msky130_fd_pr__nfet_01v8[id]
* MOSFET source currents
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
.save @m.xm12.msky130_fd_pr__nfet_01v8[is]
.save @m.xm13.msky130_fd_pr__nfet_01v8[is]
.save @m.xm14.msky130_fd_pr__nfet_01v8[is]
.save @m.xm15.msky130_fd_pr__nfet_01v8[is]
.save @m.xm16.msky130_fd_pr__nfet_01v8[is]
.save @m.xm17.msky130_fd_pr__nfet_01v8[is]
.save @m.xm18.msky130_fd_pr__nfet_01v8[is]
.save @m.xm19.msky130_fd_pr__nfet_01v8[is]
.save @m.xm20.msky130_fd_pr__nfet_01v8[is]
.save @m.xm21.msky130_fd_pr__nfet_01v8[is]
.save @m.xm22.msky130_fd_pr__nfet_01v8[is]
.save @m.xm23.msky130_fd_pr__nfet_01v8[is]
* MOSFET gate currents
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
.save @m.xm12.msky130_fd_pr__nfet_01v8[ig]
.save @m.xm13.msky130_fd_pr__nfet_01v8[ig]
.save @m.xm14.msky130_fd_pr__nfet_01v8[ig]
.save @m.xm15.msky130_fd_pr__nfet_01v8[ig]
.save @m.xm16.msky130_fd_pr__nfet_01v8[ig]
.save @m.xm17.msky130_fd_pr__nfet_01v8[ig]
.save @m.xm18.msky130_fd_pr__nfet_01v8[ig]
.save @m.xm19.msky130_fd_pr__nfet_01v8[ig]
.save @m.xm20.msky130_fd_pr__nfet_01v8[ig]
.save @m.xm21.msky130_fd_pr__nfet_01v8[ig]
.save @m.xm22.msky130_fd_pr__nfet_01v8[ig]
.save @m.xm23.msky130_fd_pr__nfet_01v8[ig]
* MOSFET bulk currents
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
.save @m.xm12.msky130_fd_pr__nfet_01v8[ib]
.save @m.xm13.msky130_fd_pr__nfet_01v8[ib]
.save @m.xm14.msky130_fd_pr__nfet_01v8[ib]
.save @m.xm15.msky130_fd_pr__nfet_01v8[ib]
.save @m.xm16.msky130_fd_pr__nfet_01v8[ib]
.save @m.xm17.msky130_fd_pr__nfet_01v8[ib]
.save @m.xm18.msky130_fd_pr__nfet_01v8[ib]
.save @m.xm19.msky130_fd_pr__nfet_01v8[ib]
.save @m.xm20.msky130_fd_pr__nfet_01v8[ib]
.save @m.xm21.msky130_fd_pr__nfet_01v8[ib]
.save @m.xm22.msky130_fd_pr__nfet_01v8[ib]
.save @m.xm23.msky130_fd_pr__nfet_01v8[ib]
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
.save @m.xm21.msky130_fd_pr__nfet_01v8[vgs] @m.xm21.msky130_fd_pr__nfet_01v8[vds] @m.xm21.msky130_fd_pr__nfet_01v8[vdsat] @m.xm21.msky130_fd_pr__nfet_01v8[vth]
.save @m.xm22.msky130_fd_pr__nfet_01v8[vgs] @m.xm22.msky130_fd_pr__nfet_01v8[vds] @m.xm22.msky130_fd_pr__nfet_01v8[vdsat] @m.xm22.msky130_fd_pr__nfet_01v8[vth]
.save @m.xm23.msky130_fd_pr__nfet_01v8[vgs] @m.xm23.msky130_fd_pr__nfet_01v8[vds] @m.xm23.msky130_fd_pr__nfet_01v8[vdsat] @m.xm23.msky130_fd_pr__nfet_01v8[vth]
* Source/resistor currents
.save @r0[i]
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

* =========================
* ANALYSIS
* =========================

.op
.ac dec 100 1 10G

.end
