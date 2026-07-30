/**
 * Route guard.
 *
 * This is a *convenience*, not a security boundary. Anyone can edit client
 * state in devtools and render whatever they like. The actual enforcement is
 * `Depends(get_current_user)` on the server, which is why every protected
 * endpoint declares it independently.
 */

import { Navigate, Outlet, useLocation } from 'react-router-dom';
import { useAuth } from './AuthContext';

export function ProtectedRoute() {
  const { status } = useAuth();
  const location = useLocation();

  // Critical: while the session is being restored from the refresh cookie we
  // do not yet know whether the user is authenticated. Redirecting here would
  // bounce a logged-in user to /login on every page refresh.
  if (status === 'loading') {
    return (
      <div className="flex min-h-full items-center justify-center">
        <div className="flex items-center gap-2.5">
          {[0, 1, 2].map((index) => (
            <span
              key={index}
              className="bg-ink-faint thinking-dot size-1.5 rounded-full"
              style={{ animationDelay: `${index * 0.16}s` }}
            />
          ))}
        </div>
      </div>
    );
  }

  if (status === 'anonymous') {
    // `state` remembers where they were headed, so login can send them back.
    return <Navigate to="/login" replace state={{ from: location.pathname }} />;
  }

  return <Outlet />;
}

/** Inverse guard: keeps signed-in users off /login and /register. */
export function PublicOnlyRoute() {
  const { status } = useAuth();

  if (status === 'loading') return null;
  if (status === 'authenticated') return <Navigate to="/" replace />;

  return <Outlet />;
}
