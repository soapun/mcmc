// Learn more about F# at http://fsharp.org

open System
open System.Drawing
open MathNet
open MathNet.Numerics
open System.IO
open Angara.Filzbach
open Angara.Statistics

[<EntryPoint>]





let main argv =
    let sqr x = x * x
    let I (x, y, p1, a1, b1, p2, a2, b2, lmbda, z) =
        let k = 2. * Math.PI / lmbda
        let q (a, b) =
            k / z * Math.Sqrt(a*a*x*x + b*b*y*y)

        let q1 = q(a1, b1)
        let q2 = q(a2, b2)
        
        p1 * (sqr (a1 * b1 * SpecialFunctions.BesselJ(1., q1)/q1)) + p2 * (sqr (a2 * b2 * SpecialFunctions.BesselJ(1., q2)/q2))
    //Let lengths be measured in microns
    let z = 65000. //65 mm between particles and observation screen
    let lmbda = 0.63 //red laser

    //Let red blood cells have the same area and the same initial radius r
    
    let r = 4.0 //human red blood cells are about 4 mkm in radius

    //Bad particles are elongated at all
    
    let a1 = r
    let b1 = r
    let p1 = 0.3 //there should be a few of bad cells assuming the patient is alive
    
    let eps2 = 4. //Let normal cells be elongated, say, 2 times
    let a2 = (sqrt eps2) * r // a2 / b2 = eps2 - the elongation is eps2
    let b2 = r / (sqrt eps2) // a2 * b2 = r**2 - the area preserved is required from physics of our Couette flow
    let p2 = 1. - p1 //it is convinient to set p1+p2=1, although not strictly required 



    //Set maximum observation angle. It is an angle between straight laser beam and observation (x,y) point.
    //Usual devices has 7 degrees, however, the problem is very bad posed in this case.
    //So let's use 15 degrees and vary this parameter up and down in future.
    
    let max_angle_degree = 15.;
    let max_angle = max_angle_degree * Math.PI / 180.;
    
    //Using simple right triangle
    let x_max = z * Math.Tan(max_angle)
    let y_max = z * Math.Tan(max_angle)
    
    let n_points = 1000. // resolution of our diffraction pattern picture in both axis
    let x_ar = Generate.LinearRange(-x_max, 2.*x_max / (n_points-1.), x_max)
    let y_ar = Generate.LinearRange(-y_max, 2.*y_max / (n_points-1.), y_max)

    let height = Array.length x_ar
    let width = Array.length y_ar

    let flat2Darray array2D = 
        seq { for x in [0..(Array2D.length1 array2D) - 1] do 
                  for y in [0..(Array2D.length2 array2D) - 1] do 
                      yield array2D.[x, y] }

    let mesh_grid = flat2Darray (Array2D.init (height) (width) (fun i j -> (x_ar.[i], y_ar.[j])))
    let dif_pat = mesh_grid |> Seq.map(fun (x, y) -> I(x, y, p1, a1, b1, p2, a2, b2, lmbda, z))



    let loglike (p: Parameters) = 
        let p1 = p.GetValue("p1")
        let a1 = p.GetValue("a1")
        let b1 = p.GetValue("b1")
        let a2 = p.GetValue("a2")
        let b2 = p.GetValue("b2")
        let sigma = p.GetValue("sigma")

        let sub first second = Seq.zip first second |> Seq.map( fun (x, y) -> x - y)

        let cur_dif_pat = mesh_grid |> Seq.map(fun (x, y) -> I(x, y, p1, a1, b1, 1. - p1, a2, b2, lmbda, z))
        let error = - sqr (Seq.sum(sub dif_pat cur_dif_pat)) / (2. * sqr sigma) - float (height * width) * Math.Log(sigma * Math.Sqrt(2. * Math.PI))
        error

    let prior = 
        Parameters.Empty.
            Add("p1", Uniform(0.0, 10.0)).
            Add("a1", Uniform(0.0, 10.0)).
            Add("b1", Uniform(0.0, 10.0)).
            Add("a2", Uniform(0.0, 10.0)).
            Add("b2", Uniform(0.0, 10.0)).
            Add("sigma", Uniform(0.01, 1.0))

    let posterior = Sampler.runmcmc(prior, loglike, 0, 1)
    Sampler.print posterior
    
    0