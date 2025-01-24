import unittest
import numpy as np
from windisch.power_curve import (
    calculate_cp,
    calculate_raw_power_curve,
    apply_turbulence_effect,
    calculate_rews,
    calculate_generic_power_curve,
)


class TestWindTurbineModel(unittest.TestCase):
    def test_calculate_cp(self):
        tsr = np.array([2, 4, 6, 8, 10])
        beta = np.array([0, 0, 0, 0, 0])
        cp = calculate_cp(model="Dai et al. 2016", tsr=tsr, beta=beta)
        self.assertTrue((cp >= 0).all())  # Cp should always be non-negative

    def test_calculate_raw_power_curve(self):
        vws = np.array([5, 10, 15, 20])  # Wind speeds in m/s
        p_nom = 3000  # Nominal power in kW
        d_rotor = 126  # Rotor diameter in meters
        power_curve = calculate_raw_power_curve(
            vws=vws,
            p_nom=p_nom,
            d_rotor=d_rotor,
            model="Dai et al. 2016",
            air_density=1.225,
        )
        self.assertEqual(power_curve.shape, vws.shape)
        self.assertTrue((power_curve >= 0).all())
        self.assertTrue((power_curve <= p_nom).all())  # Power should not exceed nominal power

    def test_apply_turbulence_effect(self):
        vws = np.array([5, 10, 15, 20])
        pwt = np.array([50, 200, 500, 1000])
        ti = 0.1  # Turbulence intensity
        pwt_ti = apply_turbulence_effect(vws=vws, pwt=pwt, ti=ti, v_cutin=3, v_cutoff=25)
        self.assertEqual(pwt_ti.shape, vws.shape)
        self.assertTrue((pwt_ti >= 0).all())

    def test_calculate_rews(self):
        vws = np.array([5, 10, 15, 20])
        zhub = 100  # Hub height in meters
        d_rotor = 126  # Rotor diameter in meters
        shear = 0.2
        veer = 5  # Wind veer angle in degrees
        rews = calculate_rews(vws=vws, zhub=zhub, d_rotor=d_rotor, shear=shear, veer=veer)
        self.assertEqual(rews.shape, vws.shape)
        self.assertTrue((rews > 0).all())

    def test_calculate_generic_power_curve(self):
        vws = np.array([5, 10, 15, 20])
        p_nom = 3000  # Nominal power in kW
        d_rotor = 126  # Rotor diameter in meters
        power_curve = calculate_generic_power_curve(
            vws=vws,
            p_nom=p_nom,
            d_rotor=d_rotor,
            zhub=100,
            v_cutin=3,
            v_cutoff=25,
            ti=0.1,
            shear=0.2,
            veer=5,
            model="Dai et al. 2016",
            air_density=1.225,
        )
        self.assertEqual(power_curve.shape, vws.shape)
        self.assertTrue((power_curve >= 0).all())
        self.assertTrue((power_curve <= p_nom).all())


if __name__ == "__main__":
    unittest.main()
