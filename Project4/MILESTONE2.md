# Milestone 2 checkpoint

This checkpoint extends the verified 3D frame core to the full initial footbridge model.

## Implemented

- 15 m span, 3 m width, 10-panel Pratt-style 3D frame.
- 177 physical members, each split into two finite elements (354 FE elements total).
- 254 nodes / 1524 global DOFs.
- Shape parameterization includes truss depth `H` and parabolic top-chord `camber` for later optimization.
- D2 deck model with five longitudinal support lines at 0.75 m spacing.
- Member self-weight, 5 kPa pedestrian pressure, and 8 mm steel deck self-weight.
- D2 deck mass lumped to support-member nodes.
- Full end-line support conditions.
- Local force and von Mises stress recovery at both element ends.
- Force and moment equilibrium checks.
- D2 plate-strip hand check.
- New distributed-load / internal-force recovery benchmark.

## Initial-design results (r = 50 mm, H = 2.0 m, camber = 0)

- Structural steel mass: 3497.17 kg
- Deck mass: 2826.00 kg
- Total mass: 6323.17 kg
- Maximum deck downward deflection: 22.401 mm
- Maximum recovered von Mises stress: 173.145 MPa
- D2 plate-strip stress: 37.020 MPa (passes 53.3 MPa fatigue limit)
- Force-balance residual norm: 1.38e-8 N
- Moment-balance residual norm: 8.42e-8 N*m

The initial frame intentionally fails the structural fatigue/stress constraints, as expected for the 50 mm starting design. The next milestone should add the bridge-level vibration and linear-buckling analyses before sensitivity derivatives.
