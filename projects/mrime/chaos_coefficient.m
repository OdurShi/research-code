function Coefficient=chaos_coefficient(Max_Iter)

Chebyshev_co(1)=0.7;
Circle_co(1)=0.7;
Iterative_co(1)=0.7;
Sine_co(1)=0.7;
Sinsudal_co(1)=0.7;

for i=1:Max_Iter-1
    %Chebyshev
    Chebyshev_co(i+1)=cos(i*acos(Chebyshev_co(i)));
    % Circle
    b=0.2;a=0.5;
    Circle_co(i+1)=mod(Circle_co(i)+b-(a/(2*pi))*sin(2*pi*Circle_co(i)),1);
    % Iterative
    Iterative_co(i+1)=sin((a*pi)/Iterative_co(i));
    % Sine
    Sine_co(i+1) = sin(pi*Sine_co(i));
    % Sinsudal
    Sinsudal_co(i+1) = 2.3*Sinsudal_co(i)^2*sin(pi*Sinsudal_co(i));
end
Coefficient.Chebyshev = Chebyshev_co;
Coefficient.Circle = Circle_co;
Coefficient.Iterative = Iterative_co;
Coefficient.Sine = Sine_co;
Coefficient.Sinsudal = Sinsudal_co;
end