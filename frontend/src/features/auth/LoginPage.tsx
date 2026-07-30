import { useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from './AuthContext';
import { AuthForm, Field } from './AuthForm';

export function LoginPage() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  // Where ProtectedRoute bounced them from, so login returns them there.
  const from = (location.state as { from?: string } | null)?.from ?? '/';

  return (
    <AuthForm
      title="Welcome back"
      subtitle="Sign in to your knowledge base"
      submitLabel="Sign in"
      onSubmit={async () => {
        await login({ email, password });
        navigate(from, { replace: true });
      }}
      footer={
        <>
          Don&apos;t have an account?{' '}
          <Link to="/register" className="text-accent hover:opacity-80">
            Create one
          </Link>
        </>
      }
    >
      <Field
        label="Email"
        type="email"
        value={email}
        onChange={setEmail}
        autoComplete="email"
        required
      />
      <Field
        label="Password"
        type="password"
        value={password}
        onChange={setPassword}
        autoComplete="current-password"
        required
      />
    </AuthForm>
  );
}
