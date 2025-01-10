import torch
import torch.nn as nn


class QuadraticPolyStep(nn.Module):
    """
    regular quadratic smoothing block
        u + σ ◦ bn 1 / (a2+b2 ) (2a − A)(σ ◦ bn(r))
    """

    def __init__(self, layers, batchnorms, residual_type):
        super().__init__()
        self.A, self.alpha = layers
        self.bn1, self.bn2 = batchnorms
        self.activation = nn.ReLU()
        self.residual_type = residual_type

    def residual(self, f, u0):
        r = f - self.A(u0)

        if self.residual_type == 'regular':
            return self.activation(self.bn1(r))
        elif self.residual_type == 'bn_only':
            return self.bn1(r)
        elif self.residual_type == 'plain':
            return r
        else:
            print('{} not implemented'.format(self.residual_type))

    def forward(self, x):
        f = x[0]
        u0 = x[1]

        a, b = x[2], x[3]

        ## quadratic block
        r = self.residual(f, u0)

        u = (2 * a) / (a ** 2 + b ** 2) * r
        u -= 1 / (a ** 2 + b ** 2) * self.A(r)

        u = self.bn2(u)
        u = self.activation(u)

        return u + u0


class QuadraticPolyStep_outside(QuadraticPolyStep):
    """
    outside  (''): σ◦bn(... u + 1/(a^2+b^2) (2a − A)(σ ◦ bn(r))) # relu and bn before resolution coarsening
    """

    def forward(self, x):
        f = x[0]
        u0 = x[1]

        a, b = x[2], x[3]

        ## quadratic block
        r = self.residual(f, u0)

        u = (2 * a) / (a ** 2 + b ** 2) * r
        u -= 1 / (a ** 2 + b ** 2) * self.A(r)

        return u + u0


class QuadraticPolyStep_relu_outside_bnin(QuadraticPolyStep):
    """
        σ ◦(... u + 1/(a2+b2) bn(2a − A)(σ ◦ bn(r)))
    """

    def forward(self, x):
        f = x[0]
        u0 = x[1]

        a, b = x[2], x[3]

        ## quadratic block
        r = self.residual(f, u0)

        u = (2 * a) / (a ** 2 + b ** 2) * r
        u -= self.bn2(1 / (a ** 2 + b ** 2) * self.A(r))

        return u + u0

class QuadraticPolyStep_relu_outside(QuadraticPolyStep):
    """
        σ ◦(... u + bn 1/(a2+b2) (2a − A)(σ ◦ bn(r)))
    """

    def forward(self, x):
        f = x[0]
        u0 = x[1]

        a, b = x[2], x[3]

        ## quadratic block
        r = self.residual(f, u0)

        u = (2 * a) / (a ** 2 + b ** 2) * r
        u -= 1 / (a ** 2 + b ** 2) * self.A(r)
        u = self.bn2(u)

        return u + u0


class QuadraticPolyStep_relu_outside_bninside_out(QuadraticPolyStep):
    """
        σ ◦(... u + 1/(a2+b2) bn(2a − A)(σ ◦ bn(r)))
    """

    def forward(self, x):
        f = x[0]
        u0 = x[1]

        a, b = x[2], x[3]
        ## quadratic block

        r = self.residual(f, u0)

        u = (2 * a) / (a ** 2 + b ** 2) * r
        u -= (1 / (a ** 2 + b ** 2) * self.A(r))
        u = self.bn2(u)

        return u + u0


class QuadraticPolyStep_inside(QuadraticPolyStep):
    """
        (... u + 1/(a2+b2) (2a − σ ◦ bn A)(σ ◦ bn(r)))
    """

    def forward(self, x):
        f = x[0]
        u0 = x[1]

        a, b = x[2], x[3]

        ## quadratic block
        r = self.residual(f, u0)

        u = (2 * a) / (a ** 2 + b ** 2) * r
        u -= self.activation(self.bn2(1 / (a ** 2 + b ** 2) * self.A(r)))

        return u + u0

class QuadraticPolyStep_outbninside(QuadraticPolyStep):
    """
        (... u + 1/(a2+b2) (2a − σ ◦ bn A)(σ ◦ bn(r)))
    """

    def forward(self, x):
        f = x[0]
        u0 = x[1]

        a, b = x[2], x[3]

        ## quadratic block
        r = self.residual(f, u0)

        u = (2 * a) / (a ** 2 + b ** 2) * r
        u -= self.bn2(1 / (a ** 2 + b ** 2) * self.A(r))

        return u + u0


# ------ Linear polynomial Smoothing steps


class LinearPolyStep(nn.Module):
    def __init__(self, layers, batchnorms, regular, residual_type):
        super().__init__()
        self.A, self.B = layers
        self.bn1, self.bn2 = batchnorms
        self.activation = nn.ReLU()
        self.regular = regular
        self.residual_type = residual_type
        self.degree = 1

    def residual(self, f, u0):
        """
             regular: σ ◦ bn(f − Au)
             bn_only: ◦ bn(f − Au)
             plain:   f − Au
         """
        r = f - self.A(u0)

        if self.residual_type == 'regular':
            return self.activation(self.bn1(r))
        elif self.residual_type == 'bn_only':
            return self.bn1(r)
        elif self.residual_type == 'plain':
            return r
        else:
            return r
            #print('{} not implemented'.format(self.residual_type))

    def polynom(self, r, alpha):
        if self.regular:
            return alpha * r

        # highest degree
        if self.degree == 1:
            if len(alpha.shape) == 0:
                alpha = [alpha]
            W = torch.mul(r, alpha[0])
            W = W.type(torch.float)
            y = self.A(W)
            return y

        else:
            W = torch.mul(r, alpha[self.degree])
            y = self.A(W)

            # 2nd highest degree
            for i in range(self.degree - 1, 0, -1):
                R = torch.mul(r, alpha[i])
                y = torch.add(y, R)
                y = self.A(y)

                # constant coefficient
            R = torch.mul(r, alpha[0])
            y = torch.add(y, R)

            return y

    def forward(self, x):
        """
         classic iteration,
         *: u + σ◦bn 1/λ (σ◦bn(f − Au))"""
        f = x[0]
        u0 = x[1]

        # smoothing
        r = self.residual(f, u0)
        u = self.polynom(r, self.B)
        u = self.bn2(u)
        u = self.activation(u)

        return u + u0


class Poly_Smoother(LinearPolyStep):
    def __init__(self, layers, batchnorms, degree, regular, residual_type):
        super().__init__(layers, batchnorms, regular, residual_type)
        self.A, self.B = layers
        self.bn1, self.bn2 = batchnorms
        self.activation = nn.ReLU()
        self.degree = degree

    def forward(self, x):
        """
         classic iteration,
         *: u + σ◦bn 1/λ (σ◦bn(f − Au))"""
        f = x[0]
        u0 = x[1]

        # smoothing
        r = self.residual(f, u0)
        u = self.polynom(r, self.B)
        u = self.bn2(u)
        u = self.activation(u)

        return u + u0



class LinearPolyStep_bnrelu_outside(LinearPolyStep):

    """
    double-outside     : σ◦bn(... u + 1/λ (σ ◦ bn(f − Au)))   # bn before resolution coarsening
    double-relu-outside: σ◦(... u + bnA* 1/λ (σ ◦ bn(f − Au)))   # bn before resolution coarsening
    """
    def forward(self, x):
        """
        (**): σ◦bn(... u + 1/λ (σ ◦ bn(f − Au)))
        """
        f = x[0]
        u0 = x[1]

        # smoothing
        r = self.residual(f, u0)
        u = self.polynom(r, self.B)
        return u + u0, self.bn2


class LinearPolyStep_relu_outside(LinearPolyStep):

    def forward(self, x):
        """
        (**‡): σ(... u + bn 1/λ (σ ◦ bn(f − Au)))'''
        """
        f = x[0]
        u0 = x[1]

        # smoothing
        r = self.residual(f, u0)
        u = self.polynom(r, self.B)
        u = self.bn2(u)
        return u + u0, self.bn2

class LinearPolyStep_relu_outside_nobn(LinearPolyStep):

    def forward(self, x):
        """
        (**‡): σ(... u + bn 1/λ (σ ◦ bn(f − Au)))'''
        """
        f = x[0]
        u0 = x[1]

        # smoothing
        r = self.residual(f, u0)
        u = self.polynom(r, self.B)

        return u + u0, self.bn2
