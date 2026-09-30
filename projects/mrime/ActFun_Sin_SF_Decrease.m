function output = ActFun_Sin_SF_Decrease(maxFe, iter)
    if iter>maxFe
        output = 0;
    else
        angle = (iter*360)/maxFe;
        rate = abs(sin(deg2rad(angle)));
        output = rand<rate;
    end
end

